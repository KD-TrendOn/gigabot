FROM python:3.10.12

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt 

COPY . /app

EXPOSE 3310

CMD ["python", "bot.py", "--host", "0.0.0.0", "--port", "3310"]