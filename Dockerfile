FROM graphcore/pytorch:latest

ADD . /workspace
WORKDIR /workspace
RUN apt update \
    && apt install -y git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install -r requirements.txt \
    && git clone https://github.com/huggingface/optimum-graphcore.git

ENV PYTHONPATH "${PYTHONPATH}:/workspace/optimum-graphcore/:/workspace/optimum-graphcore/notebooks/stable_diffusion"

CMD python3 sd_gradio.py