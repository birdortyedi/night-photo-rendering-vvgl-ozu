FROM python:3.6
FROM scipy/scipy-dev:latest
FROM pytorch/pytorch

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt .
RUN python -m pip install --no-cache -r requirements.txt


COPY . /night-photo-rendering-vvgl-ozu
WORKDIR /night-photo-rendering-vvgl-ozu
