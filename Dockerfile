FROM python:3.12-slim

# O LightGBM precisa da biblioteca OpenMP.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /usr/local/bin/uv

# O ambiente fica fora de /app, que recebe volumes. UV_PYTHON_DOWNLOADS=never usa o Python da imagem.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY cli.py ./
COPY sanecast ./sanecast
COPY tests ./tests

# O usuário do contêiner não escreve em /app: os caches do numba e do pandas vão para /tmp.
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/numba HOME=/tmp

ENTRYPOINT ["python", "cli.py"]
CMD ["demo"]
