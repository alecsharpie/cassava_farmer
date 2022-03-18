FROM ubuntu:bionic

RUN mkdir .ssh
#ARG GCP_KEY
#RUN echo "$GCP_KEY" > .ssh/gcp.json
COPY credentials_gcp.json /.ssh/gcp.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/.ssh/gcp.json


#RUN echo "deb http://packages.cloud.google.com/apt gcsfuse-buster main" > /etc/apt/sources.list.d/gcsfuse.list
#RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
#RUN apt -qq update
#RUN apt -qqy install gcsfuse
RUN apt update && apt upgrade
RUN apt -y install gnupg
RUN apt -y install curl

RUN echo "deb http://packages.cloud.google.com/apt gcsfuse-bionic main" > /etc/apt/sources.list.d/gcsfuse.list
##RUN export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
#RUN echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" > /etc/apt/sources.list.d/gcsfuse.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
#RUN apt-get update
RUN apt-get install gcsfuse

RUN mkdir gcs_bucket
#RUN gcsfuse --key-file=/.ssh/gcp.json --log-file=logs.txt image-datasets-alecsharpie gcs_bucket
#CMD cat $(echo $GOOGLE_APPLICATION_CREDENTIALS)

#CMD ls gcs_bucket


#echo "image-datasets-alecsharpie /gcs_bucket gcsfuse rw,noauto,user,key_file=/.ssh/gcp.json" >> /etc/fstab
