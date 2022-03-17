FROM python:3.8.12-buster

COPY /Users/alecsharp/me/gcp/alecsharpie-cb079a37ff99.json /gcp.json

RUN echo "deb http://packages.cloud.google.com/apt gcsfuse-buster main" > /etc/apt/sources.list.d/gcsfuse.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt -qq update
RUN apt -qq install gcsfuse

RUN mkdir gcs_bucket
RUN gcsfuse --implicit-dirs image-datasets-alecsharpie gcs_bucket

CMD ls gcs_bucket
