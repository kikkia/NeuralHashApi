FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
LABEL maintainer="Kikkia <j.bezos@amazon.com>"
COPY . /app
WORKDIR /app
RUN pip3 install -r ./requirements.txt