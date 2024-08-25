FROM python:3.11

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENTRYPOINT [ "python", "prepare_knowledge_graph.py", "--preprocess_raw_dataset", "--construct_kg", "--index_kg" ]