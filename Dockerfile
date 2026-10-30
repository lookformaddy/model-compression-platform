FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY src ./src
COPY examples ./examples
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .
CMD ["python3", "examples/run_demo.py"]
