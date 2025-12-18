# Instructions

import os
from openai import OpenAI

from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.trace import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.trace import TracerProvider

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
PHOENIX_COLLECTOR_ENDPOINT = "http://localhost:6006"

PROJECT_NAME = "tracing-agent"
tracer_provider = register(
    project_name=PROJECT_NAME,
    auto_instrument=True,  # Auto-instruments OpenAI, LangChain, etc.
    batch=False,  # Send spans immediately (recommended for local dev)
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
)
print(response.choices[0].message.content)
