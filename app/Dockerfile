FROM python:3.11-slim

# This Dockerfile expects the *repo root* as the build context.
WORKDIR /srv

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY app/ /srv/app/
COPY common/ /srv/common/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
