FROM python:3.10-slim

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copie du code
COPY . .

# Port
EXPOSE 8080

# Point d'entrée
CMD ["python", "-m", "uvicorn", "app.backend:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
