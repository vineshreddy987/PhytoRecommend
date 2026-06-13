# Use lightweight Python 3.11 image
FROM python:3.11-slim

# Prevent Python buffering
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements first (layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose port
EXPOSE 5000

# Start with gunicorn (production server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "webapp.app:app"]
