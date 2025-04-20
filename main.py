# this is the python code for the jupyter notebook code

import os
import json
import torch
import numpy as np
import pandas as pd
from getpass import getpass
from IPython.display import display, Markdown
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

class Config:
    JSON_DIR = '/kaggle/input/bhagavad-gita-json'  # Dataset path
    TOP_K = 3
    EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    GEMINI_MODEL = 'gemini-1.5-flash'
    BATCH_SIZE = 128  # Optimized for Kaggle GPU
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    PREFERRED_TRANSLATORS = ['swami sivananda', 'swami ramsukhdas', 'swami gambirananda']

config = Config()

def load_gita_data():
    """Load and process all 18 chapter JSON files with the correct structure"""
    verses = []
    for chapter in range(1, 19):
        file_path = f"{config.JSON_DIR}/bhagavad_gita_chapter_{chapter}.json"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chapter_data = json.load(f)
                for verse in chapter_data['verses']:
                    if verse['translator'].lower() in config.PREFERRED_TRANSLATORS:
                        verses.append({
                            'chapter': chapter,
                            'verse_number': verse['verse_number'],
                            'text': verse['text'],
                            'translation': verse['translation'],
                            'translator': verse['translator']
                        })
        except FileNotFoundError:
            print(f"Chapter {chapter} not found.")
    return pd.DataFrame(verses)

def embed_verses(verses, model):
    embeddings = model.encode(verses.tolist(), batch_size=config.BATCH_SIZE, convert_to_tensor=True)
    return embeddings

def find_relevant_verses(user_query, verses_df, verse_embeddings, model):
    query_embedding = model.encode([user_query], convert_to_tensor=True)
    similarities = cosine_similarity(query_embedding.cpu(), verse_embeddings.cpu())[0]
    top_indices = np.argsort(similarities)[-config.TOP_K:][::-1]
    return verses_df.iloc[top_indices], similarities[top_indices]

def display_results(results_df, scores):
    for i, (index, row) in enumerate(results_df.iterrows()):
        display(Markdown(f"### Result {i+1}"))
        display(Markdown(f"**Chapter {row['chapter']}, Verse {row['verse_number']}**"))
        display(Markdown(f"> {row['text']}"))
        display(Markdown(f"**Translation by {row['translator']}**: {row['translation']}"))
        display(Markdown(f"**Similarity Score**: {scores[i]:.4f}"))

def main():
    # Load model and data
    print("🔄 Loading model and Bhagavad Gita data...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    df = load_gita_data()
    verse_embeddings = embed_verses(df['translation'], model)
    print("✅ Setup complete! Ready to answer your questions.\n")

    # Query loop
    while True:
        query = input("\n🧘 Enter your real-life problem (or type 'exit' to quit):\n> ")
        if query.lower() == 'exit':
            print("\n🕉️ Thank you for exploring wisdom from the Bhagavad Gita.")
            break

        top_verses, top_scores = find_relevant_verses(query, df, verse_embeddings, model)
        display_results(top_verses, top_scores)

if __name__ == "__main__":
    main()
