FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend /app/backend
COPY models /app/models

ENV PYTHONPATH=/app
ENV USE_ONNX=1

EXPOSE 8000

# Run the FastAPI Socket.IO server
CMD ["uvicorn", "backend.app.main:socket_app", "--host", "0.0.0.0", "--port", "8000"]
