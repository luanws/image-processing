FROM continuumio/anaconda3:latest

# Atualizar pacotes do sistema
RUN apt-get update && apt-get upgrade -y

# Instalar dependências adicionais necessárias para o OpenCV
RUN apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx

# Instalar o OpenCV no ambiente Anaconda
RUN pip install opencv-python opencv-contrib-python

# Configurar o diretório de trabalho
WORKDIR /workspaces/image-processing

# Definir o comando padrão
CMD ["bash"]
