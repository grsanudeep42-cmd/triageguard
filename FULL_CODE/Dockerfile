FROM python:3.12-slim

# Install system dependencies required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

RUN chmod +x run_demo.sh

ENV PORT=8080
ENV HOST=0.0.0.0

EXPOSE 8080

CMD ["./run_demo.sh"]
