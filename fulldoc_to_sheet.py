# fulldoc_to_sheet.py
# for fully automated autobiographical interview scoring (using Klus et al., 2025 model)
# takes sempre "edited" and pulls out the text from "free recall" and beyond. puts in spreadsheet


import os
import pandas as pd
from tqdm import tqdm 
from docx import Document


def extract_memory_docx(filepath):
    doc = Document(filepath)
    full_text = "\n".join([para.text for para in doc.paragraphs])
    if "Free Recall" in full_text:
        return full_text.split("Free Recall", 1)[1].strip()
    else:
        return None

# Folder with the documents (eg: quality checked transcripts, 1 per document)
folder_path = "Transcripts"

# List to hold results
data = []


# Go through each file in the folder
for filename in tqdm(os.listdir(folder_path), desc="Processing files"):
    if filename.endswith(".docx"):
        filepath = os.path.join(folder_path, filename)
        memory_text = extract_memory_docx(filepath)
        if memory_text:
            participant_id = os.path.splitext(filename)[0]  # Remove file extension
            data.append({"ParticipantID": participant_id, "memory": memory_text})


# Save to CSV
df = pd.DataFrame(data)
df.to_csv("memories.csv", index=False)
