# Use Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files
COPY ./app ./app
COPY ./requirements.txt .
COPY ./train_model.py .
COPY ./AB_NYC_2019.csv .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    python train_model.py

# Run the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]