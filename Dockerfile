# Usa uma imagem leve do Python
FROM python:3.10-slim

# 1. Instala o FFmpeg (Obrigatório para o Whisper)
# O git é necessário porque o whisper as vezes instala dependências direto do GitHub
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Copia apenas os requisitos primeiro (aproveita o cache do Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copia o restante do código
COPY . .

# 4. Define o modelo como 'tiny' para tentar sobreviver aos 512MB de RAM
ENV WHISPER_MODEL=tiny
# Porta padrão que o Render espera
EXPOSE 10000

# 5. Inicia o servidor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
