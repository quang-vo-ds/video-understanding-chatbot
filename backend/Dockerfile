ARG BASE_IMAGE=pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

FROM ${BASE_IMAGE}
# Set environment variable to prevent interactive prompts during the build process
ENV DEBIAN_FRONTEND=noninteractive
# Install libaio-dev and wget
RUN apt-get update && apt-get install wget libaio-dev espeak-ng ffmpeg gcc g++ -y --no-install-recommends

WORKDIR /module

# Copy the requirements file and install other dependencies
COPY ./requirements.txt ./

ENV CUDA_DOCKER_ARCH=all

RUN pip install -r requirements.txt

WORKDIR /module/src
COPY ./src /module/src
# Reset the DEBIAN_FRONTEND environment variable to avoid any unintended effects
ENV DEBIAN_FRONTEND=