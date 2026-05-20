FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.12_pytorch2
USER root
RUN apt update && apt install -y curl
USER 1000
# Install additional packages
RUN pip install torchaudio soundfile


# podman build --no-cache -t gcr.io/viewo-g/piper/agent/cpu/datasets-torch:0.0.4 .