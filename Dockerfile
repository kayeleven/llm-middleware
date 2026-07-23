# syntax=docker/dockerfile:1

# ==================================================================== #
# Base image.
# Must match the platform the wheels were vendored for:
#   pip download ... --platform manylinux2014_x86_64 --python-version 3.12
# python:3.12-slim is glibc/Linux x86_64, which manylinux2014 wheels target.
#
# AIR-GAP NOTE: on the download network, save this base image as a tarball:
#   docker pull python:3.12-slim
#   docker save python:3.12-slim -o python-3.12-slim.tar
# On the server:
#   docker load -i python-3.12-slim.tar
# ==================================================================== #
FROM python:3.12-slim

# -------------------------------------------------------------------- #
# Runtime environment.
# - PYTHONUNBUFFERED: flush stdout/stderr immediately so `docker logs` shows
#   output in real time (essential when you cannot attach a debugger).
# - PYTHONDONTWRITEBYTECODE: no .pyc clutter in the image.
# - PIP_NO_INDEX / PIP_FIND_LINKS: belt-and-suspenders so ANY pip invocation in
#   this build resolves ONLY from the vendored wheels, never the network.
# -------------------------------------------------------------------- #
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_INDEX=1 \
    PIP_FIND_LINKS=/wheels \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

# -------------------------------------------------------------------- #
# Offline dependency install.
# Copy the vendored wheels and requirements FIRST (before app code) so this
# expensive layer is cached and only rebuilds when dependencies change.
#
# The wheels/ directory must contain the top-level pins AND all transitive
# dependencies, produced on the download network with:
#   pip download -r requirements.txt -d ./wheels \
#     --platform manylinux2014_x86_64 --python-version 3.12 --only-binary=:all:
# -------------------------------------------------------------------- #
COPY wheels/ /wheels/
COPY requirements.txt /srv/requirements.txt

# --no-index + --find-links guarantees NO network access during install.
RUN pip install --no-index --find-links=/wheels -r /srv/requirements.txt \
    && rm -rf /wheels /root/.cache/pip

# -------------------------------------------------------------------- #
# Application code.
# Copied AFTER dependencies so code edits don't invalidate the pip layer.
# -------------------------------------------------------------------- #
COPY app/ /srv/app/

# -------------------------------------------------------------------- #
# /data: shared, persistent image-description cache (SQLite) + acronyms.csv.
# Create it and hand ownership to the non-root user so SQLite can write the DB
# and its WAL/SHM sidecar files. This directory is normally MOUNTED as a volume
# from the host (./data:/data); creating it here also lets the container run
# without a mount for smoke tests.
# -------------------------------------------------------------------- #
RUN mkdir -p /data

# -------------------------------------------------------------------- #
# Non-root user (IL5 posture). Create a dedicated user/group, own /srv and
# /data, then drop privileges. SQLite WAL mode needs WRITE access to /data.
# -------------------------------------------------------------------- #
RUN groupadd --system app && useradd --system --gid app --home-dir /srv app \
    && chown -R app:app /srv /data
USER app

# -------------------------------------------------------------------- #
# Network: the middleware listens on 8080 (OWUI's OpenAI API base URL points
# here). EXPOSE is documentation; actual publishing is in the compose file.
# -------------------------------------------------------------------- #
EXPOSE 8080

# Declare /data as a volume so its lifecycle is explicit. (The compose file
# still bind-mounts ./data:/data for host-side persistence.)
VOLUME ["/data"]

# -------------------------------------------------------------------- #
# Launch. Single uvicorn process (vLLM handles concurrency; we do not add our
# own workers/throttling). Host 0.0.0.0 so it is reachable within the network.
# -------------------------------------------------------------------- #
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]