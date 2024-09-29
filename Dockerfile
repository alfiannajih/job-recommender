FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# WORKDIR /app

# COPY requirements.txt /app/requirements.txt
# RUN pip install --no-cache-dir https://huggingface.co/alfiannajih/en_ner_job/resolve/main/en_ner_job-any-py3-none-any.whl
# RUN pip install --no-cache-dir -r requirements.txt

# RUN python -m nltk.downloader punkt_tab
# RUN python -m nltk.downloader wordnet
# RUN python -m nltk.downloader stopwords

# COPY . /app

# ENTRYPOINT [ "python", "prepare_knowledge_graph.py", "--preprocess_resume_dataset" ]
