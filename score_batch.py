# adapted from Klus et al., 2025 (and associated HuggingFace)
# this script runs Klus's model in *batches* of X memories, which reduces memory load
# original script reads in all memories at once. this one is slower but ideal for 
# computers with less memory and studies with many docs (>2k)

import gc
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import AutoPeftModelForSequenceClassification
from tqdm import tqdm
import os


access_token = "ACCESS TOKEN HERE"
# Setup environment
os.environ["HF_TOKEN"] = access_token
HF_TOKEN = os.getenv("HF_TOKEN")

def score_in_batches(texts, model, tokenizer, device, batch_size=8):
    scores = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Batch scoring"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model(**inputs)
            batch_scores = output.logits.squeeze().cpu().numpy()

        scores.extend(np.round(batch_scores).tolist())
        del inputs, output
        torch.cuda.empty_cache()
        gc.collect()

    return scores

def main():
   
    input_path = "./narratives.csv"
    save_path = "./narratives_output.csv"

    batch_size = 8  # adjust depending on preference

    narratives_df = pd.read_csv(input_path).dropna().reset_index(drop=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for detail in ["internal", "external"]:
        model_name = f"jonasklus/automated-ai-scoring-{detail}"
        output_column = f"{detail}_details"

        if output_column not in narratives_df.columns:
            narratives_df[output_column] = np.nan

        to_score = narratives_df[narratives_df[output_column].isna()].copy()
        if to_score.empty:
            print(f"✅ All rows already scored for {detail}.")
            continue

        texts = to_score["text"].tolist()

        tokenizer = AutoTokenizer.from_pretrained(model_name)
    

        model = AutoPeftModelForSequenceClassification.from_pretrained(
            model_name,
            problem_type="regression",
            num_labels=1
        ).to(device).eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

        print(f"⚙️ Scoring {len(texts)} rows for {detail} on {device}...")
        scores = score_in_batches(texts, model, tokenizer, device, batch_size)

        # Update output column
        narratives_df.loc[to_score.index, output_column] = scores
        narratives_df.to_csv(save_path, index=False)

        print(f"✅ Finished scoring {detail}.")

        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    print(f"🎉 All scoring complete. Output saved to: {save_path}")

if __name__ == "__main__":
    main()
