FROM python:3.11

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ARG DAGSHUB_KEY

RUN dvc remote modify origin --local access_key_id ${DAGSHUB_KEY}
RUN dvc remote modify origin --local secret_access_key ${DAGSHUB_KEY}
RUN dvc pull

ARG MLFLOW_TRACKING_URI
ARG MLFLOW_TRACKING_USERNAME
ARG MLFLOW_TRACKING_PASSWORD

RUN export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
RUN export MLFLOW_TRACKING_USERNAME=${MLFLOW_TRACKING_USERNAME}
RUN export MLFLOW_TRACKING_PASSWORD=${MLFLOW_TRACKING_PASSWORD}

ENTRYPOINT ["python", "main.py", "--train"]