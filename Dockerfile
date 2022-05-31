# base docker image:
# make sure you pulled the docker base image: docker pull continuumio/anaconda3:latest
FROM continuumio/anaconda3 

# label the docker image:
LABEL Name="lfbnet"  

# setting proxies if your are behind proxy companies:
# kindly refer to: https://docs.docker.com/network/proxy/

# define working directory inside the docker image:
WORKDIR /lfbnet

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Copy everything in the current directory into the docker image working directory.
# Recommended not to put medical data in the current directory!
COPY . /lfbnet

# Assume requirements.txt was in the current directory, install dependencies that require pip install:
RUN pip install --upgrade pip 
RUN pip install -r requirements.txt

# Run the main python code when the container is started:# 
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "python", "/lfbnet/test_docker.py"]
