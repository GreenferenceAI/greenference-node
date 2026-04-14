FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY server.py .

EXPOSE 8000

# Default entrypoint — pass --model at runtime
ENTRYPOINT ["python3", "server.py"]
CMD ["--model", "stabilityai/stable-diffusion-xl-base-1.0", "--host", "0.0.0.0", "--port", "8000"]
