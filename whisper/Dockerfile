FROM python:3.11-slim-bullseye

RUN apt-get clean && apt-get -y update && apt-get -y upgrade &&\
    apt-get install -y --no-install-recommends ffmpeg\
    git \
    build-essential && \
    rm -rf /var/lib/apt/lists/* && apt-get clean

WORKDIR /code

# Copy just the requirements first
COPY ./requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Increase timeout to wait for the new installation
RUN pip install --no-cache-dir -r requirements.txt --timeout 200

# Copy the rest of the application
COPY . .

EXPOSE 8001

CMD ["uvicorn", "whisper.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "3"]