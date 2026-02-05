# ------------------------
# 1. Base image (Python 3.12 slim)
# ------------------------
FROM python:3.12-slim

# ------------------------
# 2. Set working directory
# ------------------------
WORKDIR /app

# ------------------------
# 3. Copy only necessary folders
# ------------------------
COPY src/ src/
COPY api/ api/
COPY models/ models/

# ------------------------
# 4. Install minimal dependencies
# ------------------------
RUN pip install --no-cache-dir numpy flask

# ------------------------
# 5. Expose Flask port
# ------------------------
EXPOSE 5000

# ------------------------
# 6. Run Flask API on container start
# ------------------------
CMD ["python", "-m", "api.app"]
