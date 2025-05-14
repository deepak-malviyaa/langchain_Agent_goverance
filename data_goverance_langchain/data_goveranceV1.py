import os
import json
from typing import List
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import initialize_agent, AgentType
from langchain.agents import Tool
from langchain_groq import ChatGroq

# Load env vars
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Models
fast_llm = ChatGroq(temperature=0, model_name="llama3-8b-instant", api_key=GROQ_API_KEY)
smart_llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", api_key=GROQ_API_KEY)

# Tool: Extract schema
@tool
def extract_schema(_: str) -> str:
    """Extracts table and column names from a database schema."""
    engine = create_engine(DB_URL)
    inspector = inspect(engine)
    result = []
    for table in inspector.get_table_names():
        cols = [col['name'] for col in inspector.get_columns(table)]
        result.append({"table": table, "columns": cols})
    return json.dumps(result)

# Tool: Classify columns
@tool
def classify_columns(schema_json: str) -> str:
    """Classify columns as PII, Sensitive, or Public using LLM."""
    schema = json.loads(schema_json)
    prompt = "Classify columns in the schema as 'PII', 'Sensitive', or 'Public'. Return JSON like {table: {column: classification}}.\n\n"
    for table in schema:
        prompt += f"Table: {table['table']}\n"
        for col in table['columns']:
            prompt += f"  - {col}\n"
        prompt += "\n"
    messages = [
        SystemMessage(content="You are a data governance expert."),
        HumanMessage(content=prompt)
    ]
    response = smart_llm.invoke(messages)
    return response.content

# Tool: Generate masking SQL
@tool
def generate_masking_sql(classifications_json: str) -> str:
    """Generate masking SQL views for classified columns."""
    classifications = json.loads(classifications_json)
    sqls = []
    for table, cols in classifications.items():
        masked_cols = []
        for col, cls in cols.items():
            if cls == "PII":
                masked = f"CASE WHEN current_user != 'admin' THEN '***' ELSE {col} END AS {col}"
            elif cls == "Sensitive":
                masked = f"CASE WHEN current_user != 'admin' THEN 'REDACTED' ELSE {col} END AS {col}"
            else:
                masked = col
            masked_cols.append(masked)
        sql = f"CREATE OR REPLACE VIEW {table}_masked AS SELECT {', '.join(masked_cols)} FROM {table};"
        sqls.append(sql)
    return json.dumps(sqls, indent=2)

# Tool: Execute SQL views
@tool
def execute_sql_views(sqls_json: str) -> str:
    """Execute SQL view creation statements."""
    sqls: List[str] = json.loads(sqls_json)
    engine = create_engine(DB_URL)
    success, fail = [], []
    with engine.connect() as conn:
        for sql in sqls:
            try:
                conn.execute(text(sql))
                success.append(sql.splitlines()[0])
            except Exception as e:
                fail.append((sql, str(e)))
    return f"✅ Executed: {len(success)} views\n❌ Failed: {len(fail)}\n"

# Register tools
tools = [
    Tool.from_function(extract_schema),
    Tool.from_function(classify_columns),
    Tool.from_function(generate_masking_sql),
    Tool.from_function(execute_sql_views)
]

# Create agent
agent = initialize_agent(
    tools=tools,
    llm=fast_llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run agent
if __name__ == "__main__":
    print("🤖 Starting Data Governance Agent...\n")
    agent.invoke("Extract schema, classify columns, generate masking SQL, and if needed, execute it in the DB.")
