FROM python:3.11-slim

WORKDIR /app

# Install Python deps first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .
COPY static/ ./static/

# OpenShift runs containers as a random UID, not root.
# Make /app group-writable so the random UID can read/write here.
RUN chgrp -R 0 /app && chmod -R g=u /app

# Use a non-root user (OpenShift will override the UID anyway,
# but this signals our intent and works on plain Docker too)
USER 1001

ENV PORT=8080
EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "600", "--workers", "2", "app:app"]