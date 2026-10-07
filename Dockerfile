# Development environment: system libraries + uv. The Python packages are not baked
# into the image: `uv sync` installs the exact versions from uv.lock inside the container.
# Only the NVIDIA driver and the NVIDIA Container Toolkit are needed on the host
# (the PyTorch wheels ship their own CUDA libraries).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.15 /uv /uvx /bin/

# The virtual environment and uv-managed Pythons live outside the mounted repository
# (named volumes), so they never clash with an environment created on the host.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/home/appuser/.cache/uv-python \
    PATH=/opt/venv/bin:$PATH

# Same user/group ids as on the host, so files written to the repository belong to you.
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appuser \
    && useradd -m -u ${UID} -g ${GID} appuser \
    && mkdir -p /opt/venv /home/appuser/.cache \
    && chown -R appuser:appuser /opt/venv /home/appuser/.cache

USER appuser
WORKDIR /workspace
CMD ["bash"]
