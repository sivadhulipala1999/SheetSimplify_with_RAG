# syntax=docker/dockerfile:1

FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt 
COPY . .
CMD ["python", "api/app.py"]
EXPOSE 8000 
# EXPOSE exposes the port only in case of docker -p and not with -P