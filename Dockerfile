FROM graphcore/pytorch:latest

ADD . /workspace
WORKDIR /workspace
RUN apt update && apt install git
RUN pip install -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/workspace/optimum-graphcore/:/workspace/optimum-graphcore/notebooks/stable_diffusion"


CMD python3 sd_gradio.py