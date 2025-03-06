FROM python:3.10.5

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt


CMD ["python", "app/main.py"]