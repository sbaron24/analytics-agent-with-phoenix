import os
from io import StringIO
from openai import OpenAI
import pandas as pd
import matplotlib.pyplot as plt
import json
import duckdb
from pydantic import BaseModel, Field
from IPython.display import Markdown

from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.trace import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.trace import TracerProvider


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
PHOENIX_COLLECTOR_ENDPOINT = "http://localhost:6006"

PROJECT_NAME = "analytics_agent"
tracer_provider = register(
    project_name=PROJECT_NAME,
    auto_instrument=True,  # Auto-instruments OpenAI, LangChain, etc.
    batch=False,  # Send spans immediately (recommended for local dev)
)

# traces any call to OpenAI automagically
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
tracer = tracer_provider.get_tracer(__name__)  # for tracing manual tool calls

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


@tracer.tool()
def lookup_sales_data(prompt: str) -> str:
    """Implementation of sales data lookup from parquet file using SQL"""

    try:
        # define the table name
        table_name = "sales"

        # step 1: read the parquet file into a DuckDB table
        df = pd.read_parquet(TRANSACTION_DATA_FILE_PATH)
        duckdb.sql(
            f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df")

        # step 2: generate the SQL code

        sql_query = generate_sql_query(prompt, df.columns, table_name)
        print(sql_query)
        # clean the response to make sure it only includes the SQL code
        sql_query = sql_query.strip()
        sql_query = sql_query.replace("```sql", "").replace("```", "")

        # step 3 execute the SQL query
        with tracer.start_as_current_span("execute_sql_query", openinference_span_kind="chain") as span:
            span.set_input(sql_query)
            result = duckdb.sql(sql_query).df()
            span.set_output(value=str(result))
            span.set_status(StatusCode.OK)
        return result.to_string()
    except Exception as e:
        return f"Error accessing data: {str(e)}"


DATA_ANALYSIS_PROMPT = """
Analyze the following data: {data}
Your job is to answer the following question: {prompt}
"""


@tracer.tool()
def analyze_sales_data(prompt: str, data: str) -> str:
    """Implementation of AI-powered sales data analysis"""
    formatted_prompt = DATA_ANALYSIS_PROMPT.format(data=data, prompt=prompt)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}]
    )

    analysis = response.choices[0].message.content
    return analysis if analysis else "No alaysis could be generated"

# prompt for chart generation


CHART_CONFIGURATION_PROMPT = """
Generate a chart configuration based on this data: {data}
- Make sure there are equal length arrays when generating chart data.
The goal lis to show: {visualization_goal}
"""


class VisualizationConfig(BaseModel):
    chart_type: str = Field(..., description="Type of chart to generate")
    x_axis: str = Field(..., description="Name of the x-axis column")
    y_axis: str = Field(..., description="Name of the y-axis column")
    title: str = Field(..., description="Title of the chart")


# code for step 1 of tool 3

def extract_chart_config(data: str, visualization_goal: str) -> dict:
    """Generate chart viz config

    Args:
        data: String contraining the data to visualize
        visualization_goal: Description of what the visualization should show

    Returns:
        Dictionary containing line chart config
    """

    formatted_prompt = CHART_CONFIGURATION_PROMPT.format(
        data=data, visualization_goal=visualization_goal)

    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
        response_format=VisualizationConfig
    )

    try:
        # Extract axis and title info from response
        content = response.choices[0].message.content
        # Return structured chart config
        return {
            "chart_type": content.chart_type,
            "x_axis": content.x_axis,
            "y_axis": content.y_axis,
            "title": content.title,
            "data": data
        }
    except Exception:
        return {
            "chart_type": "line",
            "x_axis": "date",
            "y_axis": "value",
            "title": visualization_goal,
            "data": data
        }


# step 2 of chart gen
CREATE_CHART_PROMPT = """
Write python code to create a chart based on the following configuration.
Only return the code, no other text.

Requirements:
- Use `from io import StringIO` (not pd.compat.StringIO)
- Use modern pandas API
- Include all necessary import

config: {config}
"""


def create_chart(config: dict) -> str:
    """Create a chart based on the config"""
    formatted_prompt = CREATE_CHART_PROMPT.format(config=config)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}]
    )

    code = response.choices[0].message.content
    code = code.replace("```python", "").replace("```", "")
    code = code.strip()

    return code

# combine chart steps


@tracer.tool()
def generate_visualization(data: str, visualization_goal: str) -> str:
    """Generate a visualization based on the data and goal"""
    config = extract_chart_config(data, visualization_goal)
    code = create_chart(config)

    return code


tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_sales_data",
            "description": "Look up data from Stores Sales Price Elasticity Promortions Data",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "A plain english description of the DuckDB SQL the users wants generated"}
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_sales_data",
            "description": "Analyze sales data to extract insights",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "The lookup_sales_data result"},
                    "prompt": {"type": "string", "description": "A question from the user about the kind of analysis they want"}
                },
                "required": ["data", "prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_visualization",
            "description": "Generate Python code to create data visualization",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "The lookup_sales_data result"},
                    "visualization_goal": {"type": "string", "description": "A goal for what the data visualization should show from the data"}
                },
                "required": ["data", "visualization_goal"]
            }
        }
    }
]

# diction mapping function names to their implementations
tool_implementations = {
    "lookup_sales_data": lookup_sales_data,
    "analyze_sales_data": analyze_sales_data,
    "generate_visualization": generate_visualization
}


def run_agent(messages):

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
        print("Messages BEFORE system prompt: ", messages)

    if not any(
        isinstance(message, dict) and message.get('role') == "system" for message in messages
    ):
        system_prompt = {"role": "system", "content": SYSTEM_PROMPT}
        messages.append(system_prompt)
    while True:
        # Router Span
        print("Starting router call span")
        with tracer.start_as_current_span(
            "router_call",
            openinference_span_kind="chain",
        ) as span:
            span.set_input(value=messages)

            print("Making router call to OpenAI")
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
            )
            messages.append(response.choices[0].message)
            tool_calls = response.choices[0].message.tool_calls
            print("Received response with tool calls:", bool(tool_calls))
            span.set_status(StatusCode.OK)

            if tool_calls:
                print("Processing tool calls")
                messages = handle_tool_calls(tool_calls, messages)
                span.set_output(value=tool_calls)
            else:
                print("No tool calls, returning final response")
                span.set_output(value=response.choices[0].message.content)
                return response.choices[0].message.content


SYSTEM_PROMPT = """
You are a helpful assistent that can answer questions about the Store Sales Data
"""

# Decorator below makes entire handle_tool_calls call a span in Phoenix
# takes inputs to method as input, and return value as output to span


@tracer.chain()
def handle_tool_calls(tool_calls, messages):

    for tool_call in tool_calls:
        function = tool_implementations[tool_call.function.name]
        function_args = json.loads(tool_call.function.arguments)
        result = function(**function_args)
        # breakpoint()
        messages.append({"role": "tool", "content": result,
                        "tool_call_id": tool_call.id})
        # breakpoint()

    return messages


def start_main_span(messages):
    print("Starting main span with messages:", messages)

    with tracer.start_as_current_span("AgentRun", openinference_span_kind="agent") as span:
        span.set_input(value=messages)
        ret = run_agent(messages)
        print("Main span completed with return value:", ret)
        span.set_output(value=ret)
        span.set_status(StatusCode.OK)
        return ret


if __name__ == "__main__":
    # example_data = lookup_sales_data(
    #     "give me counts for the products in the latest month of the data")
    # print(example_data)
    # # analysis = analyze_sales_data(
    # #     prompt="what trends do you see in the data?", data=example_data)
    # # print(analysis)
    # viz_code = generate_visualization(
    #     example_data, "Give me a bar-chart of monthly sales for Store 1")
    # print(viz_code)
    # exec(viz_code)
    # result = run_agent(
    #     "Generate code for a chart which displays a histogram of counts for all products sold in November 2024, and share trends."
    # )
    # print(result)
    result = start_main_span(
        [{"role": "user", "content": "Which stores did the best in 2024?"}])
