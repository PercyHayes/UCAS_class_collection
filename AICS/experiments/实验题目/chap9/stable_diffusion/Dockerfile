ARG FROM_IMAGE_NAME=yellow.hub.cambricon.com/pytorch/pytorch:v1.15.0-torch1.9-ubuntu18.04-py37
FROM ${FROM_IMAGE_NAME}

RUN mkdir -p /workspace/stable_diffusion/

COPY . /workspace/stable_diffusion/
COPY pip.conf /root/.config/pip/
WORKDIR /workspace/stable_diffusion

ENV VIRTUAL_ENV=/torch/venv3/pytorch
RUN python3 -m venv $VIRTUAL_ENV \
    && rm -rf /workspace/cair_modelzoo \
    && pip3 install -r /workspace/stable_diffusion/requirements.txt
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
