FROM --platform=linux/amd64 python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .
RUN which covvfit

ENTRYPOINT ["covvfit"]

