FROM python:3.11

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
RUN pip install -r requirements.txt

CMD ["python", "main.py", "--train"]