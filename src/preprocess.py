import pandas as pd
import re
import os

def clean_text(text):
    """Basic text cleaning for complaint narratives."""
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove boilerplate like "I am writing to file a complaint..."
    text = re.sub(r"i am writing to file a complaint regarding|to whom it may concern", "", text)
    # Remove special characters and numbers (optional, depending on preference)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_complaints(input_path, output_path):
    print("Loading dataset...")
    df = pd.read_csv(input_path)
    
    # 1. Filter for the five specific products
    # Note: CFPB product names might vary slightly, common mapping below:
    target_products = [
        'Credit card', 'Credit card or prepaid card',
        'Personal loan', 'Payday loan, title loan, or personal loan',
        'Savings account', 'Checking or savings account',
        'Money transfer', 'Money transfer, virtual currency, or money service'
    ]
    
    df_filtered = df[df['Product'].isin(target_products)].copy()
    
    # 2. Remove empty narratives
    initial_count = len(df_filtered)
    df_filtered = df_filtered.dropna(subset=['Consumer complaint narrative'])
    print(f"Removed {initial_count - len(df_filtered)} rows with empty narratives.")
    
    # 3. Clean the text
    print("Cleaning narratives...")
    df_filtered['cleaned_narrative'] = df_filtered['Consumer complaint narrative'].apply(clean_text)
    
    # 4. Save the processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_filtered.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    return df_filtered

if __name__ == "__main__":
    # Update these paths to where your raw data is located
    RAW_DATA_PATH = "data/raw/complaints.csv" 
    PROCESSED_DATA_PATH = "data/processed/filtered_complaints.csv"
    
    if os.path.exists(RAW_DATA_PATH):
        preprocess_complaints(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    else:
        print(f"Error: Raw data not found at {RAW_DATA_PATH}")