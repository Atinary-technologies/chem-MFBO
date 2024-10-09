# Build stage
FROM python:3.9-slim as build


# Update the repository sources
RUN apt-get update -y && apt-get -y install \
    apt-utils \
    gcc \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#Define workdir as app
WORKDIR /app

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy the files
COPY . .

# Upgrade pip, install wheel, the submodule, and the application in one layer
# TODO: check uv again to see if we can remove package names
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install wheel && \
    pip install .

# Docker base
FROM python:3.9-slim as base

# org.opencontainers.image labels
LABEL maintainer="Atinary Technologies Sarl" \
      org.opencontainers.image.authors="Atinary Technologies Sarl <vsabanzagil@atinary.com>" \
      org.opencontainers.image.vendor="Atinary Technologies Sarl" \
      org.opencontainers.image.licenses="https://home.atinary.com/license_sdlabs" \
      org.opencontainers.image.description="Publisher docker image"

# Define workdir as app
WORKDIR /app

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Install our packages from build image
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy python packages
COPY --from=build  $VIRTUAL_ENV $VIRTUAL_ENV

ARG UID=1004
ARG GID=1004

# Create the dev group and user
RUN groupadd --gid ${GID} atinary-dev
RUN useradd --uid ${UID} -ms /bin/bash --gid atinary-dev -m atinary-dev

RUN chown -R atinary-dev:atinary-dev $VIRTUAL_ENV

# Give ownership of /app directory to the dev user
RUN chown -R atinary-dev:atinary-dev /app

# Install git
RUN apt-get -y update
RUN apt-get -y install git

# Install sudo and add the dev user to the sudoers group and disable password prompt
RUN apt-get -y install sudo
RUN usermod -aG sudo atinary-dev
RUN echo "atinary-dev ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
RUN sudo apt-get install libxrender1

USER atinary-dev
