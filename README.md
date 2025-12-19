# analytics-agent-with-phoenix

Analytics agent (based on DeepLearning course) which generate/run SQL, produce code for chart visualization, and communicate insights from data

## Prerequisites

Install [Python 3](https://www.python.org/downloads/) and [pip](https://pip.pypa.io/en/stable/installation/)

## Installation

Clone repo and install dependencies:

```bash
git clone https://github.com/sbaron24/analytics-agent-with-phoenix.git
python3 -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## Running Phoenix

Open a terminal and run:

```bash
> phoenix serve
```

In another terminal run:

```
> python3 analytics_agent.py
```

Phoenix application will be available at `http://localhost:6006`

(Can be modified in `analytics_agent.py`)

## Tools

- [Phoenix](https://arize.com/) for Tracing and Evals
