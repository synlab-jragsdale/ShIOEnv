# Start with the official Ubuntu 22.04 base image
FROM ubuntu:22.04

# Prevent interactive prompts and configure the container
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
         && \
    rm -rf /var/lib/apt/lists/*

# Copy the home.tar.gz archive into the container
COPY docker/home.tar /tmp/

# Extract the archive into /home
#
# NOTE:
# 1) If your archive is structured such that `home.tar.gz` already
#    contains a top-level "home" folder, you might do:
#    RUN tar -xzf /tmp/home.tar.gz -C /
#
# 2) Otherwise, if you have user-specific content directly in the tar,
#    extract them to /home directly:
#    RUN tar -xzf /tmp/home.tar.gz -C /home
#
RUN tar -xf /tmp/home.tar -C / \
    && rm /tmp/home.tar

ENV HOME=/home/ubuntu
