FROM python:3.12-slim


# Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3


# Working Directory
WORKDIR /app


# System Dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*


# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"


# Copy Project Files
COPY . .


# Install Dependencies
RUN uv sync


# Expose FastAPI Port
EXPOSE 8000

# Start FastAPI Application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]