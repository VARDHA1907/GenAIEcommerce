import pandas as pd  
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os
import json
import re
from tqdm import tqdm
tqdm.pandas()

#  Load CSV Data
customers = pd.read_csv("customer_data_collection.csv")
products = pd.read_csv("product_recommendation_data.csv")
customers = customers.head(5)
products = products.head(10)

# Generate Text Summaries
def summarize_customer(row):
    return (
        f"{row['Gender']} customer from {row['Location']}, aged {row['Age']}, "
        f"browsed categories: {row['Browsing_History']}, purchased: {row['Purchase_History']}, "
        f"segment: {row['Customer_Segment']}, order value: {row['Avg_Order_Value']}, "
        f"shopping during {row['Season']} (Holiday: {row['Holiday']})"
    )

def summarize_product(row):
    return (
        f"{row['Category']} > {row['Subcategory']} from {row['Brand']}, priced at {row['Price']}, "
        f"rated {row['Product_Rating']}/5, sentiment score {row['Customer_Review_Sentiment_Score']}, "
        f"popular in {row['Geographical_Location']} during {row['Season']}, related to: {row['Similar_Product_List']}"
    )

customers["Summary"] = customers.apply(summarize_customer, axis=1)
products["Summary"] = products.apply(summarize_product, axis=1)

# Generate Embeddings using Ollama + Cache
def get_ollama_embedding(text, model="nomic-embed-text", cache_dir="embedding_cache"):
    os.makedirs(cache_dir, exist_ok=True)

    # Sanitize filename key
    key = re.sub(r'[^\w\-_.]', '_', text)[:100]  # Limit to 100 chars
    path = os.path.join(cache_dir, f"{key}.json")

    # Load from cache if exists
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)

    # Else generate embedding from Ollama
    url = "http://localhost:11434/api/embeddings"
    payload = {"model": model, "prompt": text}
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        embedding = response.json()["embedding"]
        with open(path, "w") as f:
            json.dump(embedding, f)
        return embedding
    else:
        print(f"Embedding error: {response.text}")
        return [0.0] * 768  # fallback size for nomic-embed-text

# Generate embeddings
print("Generating embeddings from Ollama (with caching)...")
customers["Embedding"] = customers["Summary"].progress_apply(get_ollama_embedding)
products["Embedding"] = products["Summary"].progress_apply(get_ollama_embedding)

# Convert product embeddings to array
product_embeddings = np.array(products["Embedding"].tolist())
product_ids = products["Product_ID"].tolist()

# Recommend Top-N Products for Each Customer
def recommend(customer_embedding, top_k=3):
    similarities = cosine_similarity([customer_embedding], product_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [product_ids[i] for i in top_indices]

# Generate recommendations
print("\nGenerating product recommendations for customers:")
output = []
for i, row in customers.iterrows():
    top_products = recommend(row["Embedding"])
    print(f"Customer {row['Customer_ID']} ➜ {top_products}")
    output.append({
        "Customer_ID": row["Customer_ID"],
        "Top_Products": top_products
    })


#  Save Output

pd.DataFrame(output).to_csv("recommendation_output.csv", index=False)
print("\n Recommendations saved to 'recommendation_output.csv'")
