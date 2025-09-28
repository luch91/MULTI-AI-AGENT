FROM jenkins/jenkins:lts

USER root

# Install prerequisites + docker.io from Debian repo
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        lsb-release \
        docker.io && \
    rm -rf /var/lib/apt/lists/*

# Add Jenkins user to Docker group
RUN groupadd -f docker && usermod -aG docker jenkins

VOLUME /var/lib/docker

USER jenkins
