FROM nvcr.io/nvidia/pytorch:22.12-py3
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH /root/.cargo/bin:$PATH

RUN pip install maturin
WORKDIR char-counter
COPY char-counter .
RUN maturin build --release && pip install .

RUN pip install transformers pytorch-lightning \
    torchmetrics pandas jsonargparse[signatures] datasets \
    flair psycopg2-binary optuna \
    zstandard

RUN pip install tensorboard 

WORKDIR workdir
COPY scripts .

CMD ["/usr/sbin/sshd", "-D"]