## Parent image
FROM python:3.10-slim

## Essential environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

## Work directory inside the Docker container
WORKDIR /app

## Installing system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

## Copying all contents from local to app
COPY . .

## Run setup.py
RUN pip install --no-cache-dir -e .

# Used PORTS
EXPOSE 8501
EXPOSE 9999

# Run the app 
CMD ["python", "app/main.py"]