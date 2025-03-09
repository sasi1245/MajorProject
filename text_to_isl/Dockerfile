# Use official Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port 5002
EXPOSE 5002

# Start FastAPI server
CMD ["uvicorn", "text_to_isl:app", "--host", "0.0.0.0", "--port", "5002"]
