FROM python:3.8.12-buster

RUN mkdir .ssh
#ARG GCP_KEY
#RUN echo "$GCP_KEY" > .ssh/gcp.json
COPY credentials_gcp.json /.ssh/gcp.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/.ssh/gcp.json


RUN echo "deb http://packages.cloud.google.com/apt gcsfuse-buster main" > /etc/apt/sources.list.d/gcsfuse.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt -qq update
RUN apt -qqy install gcsfuse

RUN mkdir gcs_bucket
RUN gcsfuse --implicit-dirs image-datasets-alecsharpie gcs_bucket

CMD ls gcs_bucket
