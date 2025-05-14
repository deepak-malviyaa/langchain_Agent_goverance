import os
import requests
from sqlalchemy import create_engine, inspect
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# Step 1: Extract schema
def extract_schema(database_url):
    engine = create_engine(database_url)
    inspector = inspect(engine)
    schema = []
    for table_name in inspector.get_table_names():
        columns = inspector.get_columns(table_name)
        schema.append({
            "table": table_name,
            "columns": [col["name"] for col in columns]
        })
    return schema

# Step 2: Build prompt
def build_prompt(schema):
    prompt = "Classify each column in terms of sensitivity (PII, sensitive, public).\n\n"
    for table in schema:
        prompt += f"Table: {table['table']}\n"
        for col in table["columns"]:
            prompt += f"  - Column: {col}\n"
        prompt += "\n"
    prompt += "Return the result as JSON."
    return prompt

# Step 3: Call Groq LLM
def call_llm(prompt, api_key):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are a data governance expert."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, headers=headers, json=body)
    return response.json()["choices"][0]["message"]["content"]

# Main function
def main():
    print("🔍 Extracting schema...")
    schema = extract_schema(DATABASE_URL)

    print("🧠 Building prompt...")
    prompt = build_prompt(schema)

    print("🤖 Querying LLM...")
    result = call_llm(prompt, GROQ_API_KEY)

    print("\n📊 Classification Result:\n")
    print(result)

if __name__ == "__main__":
    main()
