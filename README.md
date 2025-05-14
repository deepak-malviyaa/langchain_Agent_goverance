# langchain_Agent_goverance

📊 AI Agent for Data Governance with LangChain, PostgreSQL & GROQ
This project is an AI-powered agent built using LangChain, integrated with PostgreSQL and powered by the GROQ API (LLM), designed to assist with data governance tasks such as metadata analysis, data quality checks, PII identification, SQL generation, and more.

🧠 Features

✅ Natural Language to SQL Queries

✅ Analyze metadata from PostgreSQL (catalog, lineage, etc.)

✅ Identify PII data and compliance risks

✅ Auto-generate documentation for tables and columns

✅ Answer questions about your data governance rules

✅ Powered by GROQ LLM for fast and accurate responses

# You need to add this 2 in your .env file
# PostgreSQL database connection URL
DB_URL=postgresql+psycopg2://username:password@host:port/dbname

# GROQ API Key (for LLM)
GROQ_API_KEY=your_groq_api_key_here
