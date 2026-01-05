FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.10_opencv
USER root
RUN apt update && apt install -y curl
USER 1000
# Install additional packages
RUN pip install torchaudio soundfile
