FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

ARG VENV="cosyvoice"
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "-c"]

# 换 apt 源（阿里云）
RUN sed -i 's@http://.*.ubuntu.com@http://mirrors.aliyun.com@g' /etc/apt/sources.list && \
    apt-get update -y && \
    apt-get install -y git build-essential wget ffmpeg unzip git-lfs sox libsox-dev && \
    git lfs install && \
    apt-get clean

# 兼容 ONNXRuntime 对 libcudnn.so.8 的硬编码
RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.9 \
          /usr/lib/x86_64-linux-gnu/libcudnn.so.8 && \
    ldconfig

# Miniforge
COPY docker/Miniforge3-Linux-x86_64.sh /tmp/miniforge.sh
RUN bash /tmp/miniforge.sh -b -p /opt/conda && rm /tmp/miniforge.sh
ENV PATH=/opt/conda/bin:$PATH

# 换 conda 源（清华源）
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ && \
    conda config --set show_channel_urls yes && \
    conda config --set channel_priority strict

# 创建虚拟环境
RUN conda create -y -n ${VENV} python=3.10
ENV PATH=/opt/conda/envs/${VENV}/bin:$PATH
ENV CONDA_DEFAULT_ENV=${VENV}

WORKDIR /workspace
ENV PYTHONPATH=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS

# 复制源码
COPY . /workspace/CosyVoice

# 安装本地下载的torch whl文件
COPY whl/torch-2.8.0+cu129-cp310-cp310-manylinux_2_28_x86_64.whl /tmp/
COPY whl/torchaudio-2.8.0+cu129-cp310-cp310-manylinux_2_28_x86_64.whl /tmp/

# 换 pip 源（阿里云）
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip install /tmp/torch-2.8.0+cu129-cp310-cp310-manylinux_2_28_x86_64.whl \
               /tmp/torchaudio-2.8.0+cu129-cp310-cp310-manylinux_2_28_x86_64.whl && \
    rm -f /tmp/*.whl && \
    pip cache purge

# 安装依赖
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    conda install -y -n ${VENV} -c conda-forge pynini==2.1.5 && \
    pip install -r /workspace/CosyVoice/requirements.txt

# 安装 ttsfrd
USER root
RUN conda run -n ${VENV} pip install --no-cache-dir \
        /workspace/CosyVoice/pretrained_models/CosyVoice-ttsfrd/ttsfrd_dependency-0.1-py3-none-any.whl \
        /workspace/CosyVoice/pretrained_models/CosyVoice-ttsfrd/ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl && \
    cd /workspace/CosyVoice/pretrained_models/CosyVoice-ttsfrd && \
    unzip -o resource.zip && \
    rm -f resource.zip

WORKDIR /workspace/CosyVoice
CMD ["python3", "api.py"]