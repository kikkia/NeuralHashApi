FROM tiangolo/uwsgi-nginx-flask:latest
COPY ./requirements.txt /var/www/requirements.txt
RUN pip3 install -r /var/www/requirements.txt