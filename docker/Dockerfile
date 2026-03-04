FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install common utilities and networking tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        apt-utils \
        coreutils \
        iputils-ping \
        iproute2 \
        net-tools \
        dnsutils \
        curl \
        wget \
        less \
        iptables \
        ethtool \
        7zip \
        file \
        sysstat \
         && \
    rm -rf /var/lib/apt/lists/*

COPY home.tar /tmp/

RUN tar -xf /tmp/home.tar -C / \
    && rm /tmp/home.tar

ENV HOME=/home/ubuntu
