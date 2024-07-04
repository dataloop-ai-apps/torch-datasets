FROM dataloopai/dtlpy-agent:cpu.py3.8.opencv4.7

# Install additional packages
RUN pip install torchaudio soundfile