FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y gstreamer1.0-libav gstreamer1.0-plugins-bad gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-tools  python3-pip libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
RUN apt-get install gstreamer1.0-plugins-base-apps
RUN apt-get install -y cmake g++ 
RUN apt-get install -y libglib2.0-dev
RUN pip3 install --upgrade pip
#RUN pip3 install opencv-python
RUN apt-get install -y mc git
WORKDIR /playground
ENV HOME=/playground
