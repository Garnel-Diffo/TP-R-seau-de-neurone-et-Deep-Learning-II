# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose port
EXPOSE 5000

# Run with gunicorn for production-like env
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--workers", "1"]
