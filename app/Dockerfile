# Use official slim Python base
FROM python:3.11-slim

WORKDIR /srv

# install system deps
RUN apt-get update && apt-get install -y \
    gcc \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy root requirements files
COPY requirements.txt /srv/requirements.txt
COPY requirements-dashboard.txt /srv/requirements-dashboard.txt

# Install requirements
RUN pip install --no-cache-dir -r /srv/requirements.txt
RUN pip install --no-cache-dir -r /srv/requirements-dashboard.txt

# Copy app and other folders
COPY app/ /srv/app/
COPY common/ /srv/common/
COPY dashboard/ /srv/dashboard/
COPY data/ /srv/data/ 
COPY eval/ /srv/eval/
COPY tests/ /srv/tests/
COPY train/ /srv/train/
COPY infra/ /srv/infra/

# Expose port for FastAPI
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
