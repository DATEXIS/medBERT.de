# syntax=docker/dockerfile:1

FROM registry.datexis.com/bwinter/datexis-pytorch:python3.7-cuda11.0


WORKDIR /deduplicate

RUN pip install torch
RUN pip install sklearn
RUN pip install pandas
RUN pip install tensorboard
RUN pip install fire
RUN pip install matplotlib
RUN pip install pytorch-lightning==1.4.9
RUN pip install transformers==4.11.2
RUN pip install torchmetrics

COPY germeval18.py /germeval18/germeval18.py