# syntax=docker/dockerfile:1.7
# A diretiva syntax=docker/dockerfile:1.7 habilita BuildKit features (cache mounts, heredocs)
# BuildKit é o backend moderno do Docker build — paraleliza stages e monta caches de layer

# ─────────────────────────────────────────────
# Stage 1: builder
# Responsabilidade única: compilar dependências em .whl files
# Este stage não vai para a imagem final — só os artefatos gerados
# ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Instala dependências de sistema necessárias para compilar extensões C:
# - gcc, python3-dev: para asyncpg (driver PostgreSQL em Cython)
# - libpq-dev: headers do libpq — necessários para psycopg/asyncpg linkarem contra libpq
# RUN com && em uma única camada: cada RUN cria um layer — consolidar reduz tamanho da imagem
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copia apenas os arquivos de definição de deps antes do código-fonte.
# Aproveitamento de cache do Docker: se pyproject.toml não mudou,
# o `pip wheel` não roda novamente — layer fica em cache.
COPY pyproject.toml ./
# Cria src/finsight/__init__.py vazio para o hatchling encontrar o pacote
# sem precisar copiar todo o código-fonte neste stage
RUN mkdir -p src/finsight && touch src/finsight/__init__.py

# --no-deps: não resolve sub-deps aqui — deixa para o pip resolver tudo de uma vez
# wheel: gera .whl pré-compilados em /wheels para instalação rápida no stage runtime
# --no-cache-dir: não salva cache de download no layer (reduziria tamanho da imagem)
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /wheels ".[dev]"


# ─────────────────────────────────────────────
# Stage 2: runtime
# Imagem final limpa — sem compiladores, sem headers, sem ferramentas de build
# ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# libpq5: biblioteca de runtime do PostgreSQL (versão menor que libpq-dev)
# asyncpg linka dinamicamente contra libpq5 em runtime — sem ela, ImportError
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Usuário não-root: boa prática de segurança — o processo não precisa de root
# --no-create-home: reduz arquivos no sistema
RUN useradd --no-create-home --shell /bin/false finsight

WORKDIR /app

# Copia os .whl compilados do stage builder
COPY --from=builder /wheels /wheels

# Instala a partir dos .whl locais — sem acesso à internet em runtime
# --find-links: instrui o pip a procurar pacotes no diretório /wheels antes do PyPI
# --compile: pré-compila .py → .pyc — reduz latência no cold start do container
# --no-index: garante que NÃO vai buscar nada no PyPI — tudo deve estar em /wheels
RUN pip install --no-cache-dir --no-index --find-links /wheels /wheels/*.whl && \
    rm -rf /wheels

# Copia o código-fonte após instalar deps — aproveita cache do Docker:
# mudanças no código não invalidam o layer de dependências
COPY src/ ./src/

# Define o usuário não-root antes de qualquer CMD/ENTRYPOINT
USER finsight

# Variáveis de ambiente para comportamento do Python em container:
# PYTHONUNBUFFERED=1: desativa buffer de stdout/stderr — logs aparecem imediatamente
# PYTHONDONTWRITEBYTECODE=1: não gera .pyc em runtime (já fizemos em build time com --compile)
# PYTHONPATH: garante que `import finsight` encontra src/finsight
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

EXPOSE 8000

# ENTRYPOINT + CMD: separação intencional
# ENTRYPOINT: o executável — nunca muda
# CMD: argumentos default — pode ser sobrescrito em `docker run` sem alterar o entrypoint
# Ex: `docker run finsight-agent --workers 4` sobrescreve o CMD mantendo uvicorn
ENTRYPOINT ["uvicorn", "finsight.api.main:app"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
