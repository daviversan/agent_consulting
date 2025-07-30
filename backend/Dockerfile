# backend/Dockerfile
FROM python:3.11-slim

# Instala o Tesseract OCR e outras dependências de sistema
RUN apt-get update && apt-get install -y tesseract-ocr ffmpeg libsm6 libxext6

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

# Expõe a porta 8000
EXPOSE 8000

# Comando para iniciar o servidor
# A porta 10000 é a padrão que o Render espera
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]