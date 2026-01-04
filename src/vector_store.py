import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

def create_vector_db(input_csv, db_path):
    # 1. Load Data
    df = pd.read_csv(input_csv)
    
    # 2. Stratified Sampling (10k-15k)
    # This ensures we get a fair mix of all 5 products
    sample_size = min(12000, len(df))
    df_sample = df.groupby('Product', group_keys=False).apply(
        lambda x: x.sample(n=int(sample_size * len(x) / len(df)))
    )
    print(f"Sampled {len(df_sample)} complaints for indexing.")

    # 3. Text Chunking
    # We use 500 chars with 50 char overlap to keep context across splits
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )

    chunks = []
    metadata = []

    for _, row in df_sample.iterrows():
        narrative = str(row['cleaned_narrative'])
        doc_chunks = text_splitter.split_text(narrative)
        
        for i, chunk in enumerate(doc_chunks):
            chunks.append(chunk)
            metadata.append({
                "complaint_id": row.get('Complaint ID', 'N/A'),
                "product": row['Product'],
                "chunk_index": i
            })

    # 4. Initialize Embedding Model
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 5. Create and Persist Vector Store (ChromaDB)
    print("Creating vector index... this may take a few minutes.")
    vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadata,
        persist_directory=db_path
    )
    
    vector_db.persist()
    print(f"Vector database saved to {db_path}")

if __name__ == "__main__":
    PROCESSED_DATA = "data/processed/filtered_complaints.csv"
    VECTOR_DB_DIR = "vector_store/chroma_db"
    
    if os.path.exists(PROCESSED_DATA):
        create_vector_db(PROCESSED_DATA, VECTOR_DB_DIR)
    else:
        print("Error: Processed data not found. Run Task 1 first.")