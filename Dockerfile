# Utilizar una imagen base de NVIDIA con CUDA y cuDNN
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Actualizar los repositorios e instalar dependencias
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Actualizar pip
RUN pip3 install --upgrade pip

# Instalar TensorFlow compatible con CUDA 11.2
RUN pip3 install tensorflow==2.13.0
RUN pip3 install tensorflow_probability

# Establecer variables de entorno para CUDA
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH /usr/local/cuda/bin:$PATH

# Definir el punto de entrada del contenedor
ENTRYPOINT ["/bin/bash"]



