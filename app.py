import os
from openai import OpenAI
import pandas as pd
import json
import duckdb
from pydantic import BaseModel, Field
from IPython.display import Markdown

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"

# define path to tx data
TRANSACTION_DATA_FILE_PATH = 'data/Store_Sales_Price_elasticity_Promotions_Data.parquet'

# prompt template for step 2 of tool 1
SQL_GENERATION_PROMPT = """
Generate a DuckDB SQL query based on a prompt. Do not reply with anything besides the SQL query.
Use DuckDB syntax.

The prompt is: {prompt}

The available columns are: {columns}
The table name is: {table_name}
"""


def generate_sql_query(prompt: str, columns: list, table_name: str) -> str:
    """Generate a SQL query based on a prompt"""
    formatted_prompt = SQL_GENERATION_PROMPT.format(
        prompt=prompt, columns=columns, table_name=table_name
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}]
    )

    return response.choices[0].message.content


def lookup_sales_data(prompt: str) -> str:
    """Implementation of sales data lookup from parquet file using SQL"""

    try:
        # define the table name
        table_name = "sales"

        # step 1: read the parquet file into a DuckDB table
        df = pd.read_parquet(TRANSACTION_DATA_FILE_PATH)
        duckdb.sql(
            f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df")

        print(df.columns)
        # step 2: generate the SQL code
        sql_query = generate_sql_query(prompt, df.columns, table_name)
        # clean the response to make sure it only includes the SQL code
        sql_query = sql_query.strip()
        print(sql_query)
        sql_query = sql_query.replace("```sql", "").replace("```", "")

        # step 3 execute the SQL query
        result = duckdb.sql(sql_query).df()

        return result.to_string()
    except Exception as e:
        return f"Error accessing data: {str(e)}"


example_data = lookup_sales_data(
    "Give me the name of the all the tables in the database and their columns")
print(example_data)
