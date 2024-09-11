FROM dataloopai/dtlpy-agent:cpu.py3.8.opencv4.7
USER root
RUN apt update && apt install -y curl
USER 1000
# Install additional packages
RUN pip install torchaudio soundfile