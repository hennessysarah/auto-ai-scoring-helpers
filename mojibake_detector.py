
import pandas as pd
import ftfy
import os
import sys

from unidecode import unidecode


def find_and_fix_mojibake_pandas(input_filepath, output_filepath):
    """
    Finds and fixes mojibake in a CSV file using pandas,
    then prompts the user to save a clean version.

    Returns the number of cells that were fixed.
    """
    try:
        # Step 1: Read the CSV file into a pandas DataFrame.
        # Use utf-8-sig to handle files that may have a BOM.
        df = pd.read_csv(input_filepath, encoding='utf-8-sig')
        original_df = df.copy()

        # Step 2: Apply ftfy.fix_text to all string columns.
        fix_count = 0
        for col in df.columns:
            # Check if the column contains string data
            if pd.api.types.is_string_dtype(df[col]):
                # Apply the ftfy function and count the number of changes
                df[col] = df[col].astype(str).apply(ftfy.fix_text).str.replace('Ä¶', '', regex=False).apply(unidecode)
                fix_count += (df[col] != original_df[col]).sum()


        if fix_count > 0:
            print(f"✅ Found and fixed {fix_count} instances of mojibake in the file.")
        else:
            print("✅ No mojibake found in the file.")
            return 0

        # Step 3: Ask the user for confirmation before saving.
        user_input = input("Would you like to save the cleaned file? (y/n): ")
        if user_input.lower().strip() == 'y':
            # Step 4: Write the cleaned DataFrame to a new CSV file.
            df.to_csv(output_filepath, index=False, encoding='utf-8')
            print(f"✅ Successfully wrote the cleaned file to: {output_filepath}")
        else:
            print("Action cancelled. The file was not modified.")

        return fix_count

    except FileNotFoundError:
        print(f"❌ Error: File not found at {input_filepath}")
        return -1
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return -1


# paste your path here
input_file = 'memories.csv'

base, ext = os.path.splitext(input_file)
output_file = f"{base}_cleaned{ext}"


total_mojibake = find_and_fix_mojibake_pandas(input_file, output_file)

if total_mojibake > 0:
    print("\n💡 The fixed file is ready to be used with the AI scoring model.")


