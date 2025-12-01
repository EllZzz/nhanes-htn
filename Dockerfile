FROM python:3.11-slim

# Evitar .pyc y asegurar logs sin buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Dependencias del sistema (por si alguna wheel necesita compilar algo)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Copiar código del proyecto Kedro
COPY pyproject.toml* setup.cfg setup.py* .  || true
COPY src ./src
COPY conf ./conf
COPY data ./data

# Por si acaso, que Python vea "src" sin problemas
ENV PYTHONPATH=/app/src

# Entry point: usar el CLI de Kedro
ENTRYPOINT ["kedro"]

# Comando por defecto: ejecutar todos los pipelines registrados en __default__
CMD ["run"]
