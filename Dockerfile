FROM --platform=linux/amd64 ubuntu:24.04

# --- Install OS Required Packages --- #
RUN apt update && apt install -y --no-install-recommends python3 python3-venv python3-pip gcc
