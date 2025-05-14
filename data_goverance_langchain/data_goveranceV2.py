import os
import json
import re
from typing import List
from sqlalchemy import create_engine, inspect, text
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize LLMs
llm_fast = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0, api_key=GROQ_API_KEY)
llm_smart = ChatGroq(model_name="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0, api_key=GROQ_API_KEY)

# Extract DB schema
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

# Format schema for LLM
def format_schema_for_prompt(schema):
    lines = []
    for table in schema:
        lines.append(f"Table: {table['table']}")
        for col in table["columns"]:
            lines.append(f"  - {col}")
        lines.append("")  # blank line between tables
    return "\n".join(lines)

# Classify columns using LLM
def classify_columns(schema):
    prompt_template = ChatPromptTemplate.from_template(
        """
You are a data governance expert. Classify the following columns as 'PII', 'Sensitive', or 'Public'.
Return only valid JSON in the format: {{"table_name": {{"column_name": "classification"}}}}

{schema}
        """
    )
    parser = JsonOutputParser()
    chain = prompt_template | llm_smart | parser
    return chain.invoke({"schema": format_schema_for_prompt(schema)})

# Generate masking SQL using LLM
def generate_masking_sql_llm(classifications: dict) -> List[str]:
    prompt = f"""
You are a data privacy and masking expert. Based on the column classifications, generate SQL views with appropriate masking.

Rules:
- For PII: mask with '***' unless user is 'admin'
- For Sensitive: mask with 'REDACTED' unless user is 'admin'
- Public columns are shown as-is

Output format:
Only return valid CREATE OR REPLACE VIEW SQL statements, no explanation, no markdown.

Input JSON:
{json.dumps(classifications, indent=2)}
    """

    messages = [
        SystemMessage(content="You generate secure SQL views based on classification."),
        HumanMessage(content=prompt)
    ]
    response = llm_smart.invoke(messages)
    sql_block = response.content.strip()

    print("\n🧾 Raw LLM SQL Output:\n", sql_block)

    # Extract SQL views reliably
    pattern = r"(CREATE OR REPLACE VIEW .*?;)(?=\n|$)"
    sqls = re.findall(pattern, sql_block, re.DOTALL | re.IGNORECASE)

    # Fallback if nothing matched
    if not sqls and "CREATE" in sql_block.upper():
        sqls = [block.strip() + ";" for block in sql_block.split(";\n") if block.strip().upper().startswith("CREATE")]

    return sqls

# Apply views to DB
def apply_views(sql_statements: List[str]):
    engine = create_engine(DB_URL)
    success, fail = [], []

    with engine.connect() as conn:
        for sql in sql_statements:
            try:
                print("\n🛠️ Applying View SQL:\n", sql)
                with conn.begin():  # Transaction block
                    conn.execute(text(sql))
                success.append(sql.splitlines()[0])
            except Exception as e:
                print("❌ Failed to apply view:", e)
                fail.append((sql, str(e)))

    return success, fail

# Freeform Q&A
def handle_user_query(query: str) -> str:
    messages = [
        SystemMessage(content="You are a helpful data governance assistant. Only answer questions related to data governance."),
        HumanMessage(content=query)
    ]
    response = llm_fast.invoke(messages)
    return response.content

# Detect intent from query
def detect_intent(user_input: str) -> str:
    prompt = f"""
You are a command interpreter for a data governance assistant.

Classify the user message into one of these actions:
- extract_schema: Load the database schema
- classify_columns: Classify columns as PII/Sensitive/Public
- generate_masking_sql: Generate masking SQL views
- apply_views: Apply the masking views to the database
- none: If it's a general question or not a command

User message:
\"{user_input}\"

Respond with only one of: extract_schema, classify_columns, generate_masking_sql, apply_views, none.
"""
    response = llm_fast.invoke([HumanMessage(content=prompt)])
    return response.content.strip().lower()

# Main agent loop
def main():
    print("🤖 Data Governance Agent (Interactive Mode)")

    schema = None
    classifications = None
    masking_sqls = None

    while True:
        user_input = input("\n💬 Ask or instruct the agent (type 'exit' to quit):\n> ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting...")
            break

        intent = detect_intent(user_input)

        if intent == "extract_schema":
            print("🔍 Extracting database schema...")
            schema = extract_schema(DB_URL)
            print(json.dumps(schema, indent=2))

        elif intent == "classify_columns":
            if not schema:
                print("⚠️ Extract schema first before classification.")
                continue
            print("🧠 Classifying columns...")
            classifications = classify_columns(schema)
            print(json.dumps(classifications, indent=2))

        elif intent == "generate_masking_sql":
            if not classifications:
                print("⚠️ Classify columns before generating masking SQL.")
                continue
            print("🔐 Generating masking SQL views...")
            masking_sqls = generate_masking_sql_llm(classifications)
            print("\n📜 Masking SQL Statements:")
            for sql in masking_sqls:
                print(sql + "\n")

        elif intent == "apply_views":
            if not masking_sqls:
                print("⚠️ No SQL views available to apply. Generate them first.")
                continue
            print("🚀 Applying views to the database...")
            success, fail = apply_views(masking_sqls)
            print(f"\n✅ Views Created: {len(success)}\n❌ Failures: {len(fail)}")
            if fail:
                for sql, err in fail:
                    print(f"\nSQL:\n{sql}\nError: {err}")

        elif intent == "none":
            print("\n🧠 Answer:")
            print(handle_user_query(user_input))

        else:
            print(f"❓ Unrecognized intent: '{intent}'")

# Run the agent
if __name__ == "__main__":
    main()
