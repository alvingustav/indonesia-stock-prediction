FROM mcr.microsoft.com/vscode/devcontainers/miniconda:0-3

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        build-essential \
        curl \
        git \
        vim \
        wget \
        ca-certificates \
    && apt-get autoremove -y && apt-get clean -y

# Create conda environment for stock prediction
COPY requirements.txt /tmp/requirements.txt
RUN conda create -n stockpred python=3.9 -y \
    && echo "conda activate stockpred" >> ~/.bashrc

# Install Python packages
SHELL ["/bin/bash", "--login", "-c"]
RUN conda activate stockpred && pip install -r /tmp/requirements.txt

# Set working directory
WORKDIR /workspace

# Copy setup scripts
COPY scripts/ /workspace/scripts/
RUN chmod +x /workspace/scripts/*.sh

# Set default conda environment
ENV CONDA_DEFAULT_ENV=stockpred
ENV PATH=/opt/conda/envs/stockpred/bin:$PATH
