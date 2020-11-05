FROM tensorflow/tensorflow
COPY ./ .
#COPY server.py .
#COPY requirement.txt .
COPY classification_tokenizer.pkl /tmp/.

RUN ls -l .
RUN ls -l /tmp/

RUN python --version

RUN pip install --upgrade pip &&\
    pip install --no-cache-dir -r requirement.txt

CMD ["python", "server.py"]
