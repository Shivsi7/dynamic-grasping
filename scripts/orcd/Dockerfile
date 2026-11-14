# Environment only (the repo is synced separately): Python 3.12 + Drake + jax[cuda12] + sbx, venv at /opt/venv.
# Same recipe as dyngrasp_rl.def; python:3.12-bookworm already has the system libraries Drake needs.
FROM python:3.12.16-bookworm
COPY requirements.lock /opt/requirements.lock
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir -q --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir -q -r /opt/requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu
ENV PATH=/opt/venv/bin:$PATH
