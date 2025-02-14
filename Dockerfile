FROM --platform=linux/amd64 python:3.11-slim
RUN pip install --no-cache-dir covvfit==0.2.0
ENTRYPOINT ["covvfit"]

