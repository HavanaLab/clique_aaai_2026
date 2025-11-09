FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm
COPY requirements.txt .
COPY requirements-post.txt .
RUN pip3 install -r requirements.txt
# needs to be installed after torch
RUN pip3 install -r requirements-post.txt