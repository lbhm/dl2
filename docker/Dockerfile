FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends git

ADD requirements_docker.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /dl2

ENV DL2_HOME /dl2

CMD ["/bin/bash"]