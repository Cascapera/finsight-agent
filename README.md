# FinSight Agent

Sistema **multi-agente** de análise de ativos financeiros. Dada uma pergunta em
linguagem natural sobre um ticker, três agentes especializados rodam **em paralelo**
(notícias/sentimento, métricas quantitativas e RAG sobre relatórios em PDF) e um quarto
agente **sintetiza** tudo numa resposta única — transmitida ao cliente via streaming.

Construído com **LangGraph**, **pgvector**, **FastAPI/SSE**, observabilidade
(**Prometheus** + **LangSmith**) e deploy em **Fly.io**.

---

## Arquitetura

```
FastAPI (SSE)
   │
   ▼
Orchestrator (LangGraph — grafo diamante)
   │  FAN-OUT paralelo
   ├── Research Agent   → Tavily    → notícias + sentimento
   ├── Financial Agent  → yfinance  → Sharpe, VaR, retorno, volatilidade
   └── RAG Agent        → pgvector  → chunks de relatórios PDF (HyDE + re-ranking)
   │  FAN-IN
   ▼
Risk Agent (síntese final)  →  FinalAnswer
   │
   ├── Redis        (cache de respostas, TTL 1h)
   ├── LangSmith    (tracing, correlação por execution_id)
   └── Prometheus   (métricas por nó: duração, execuções, erros)
```

**Princípios de design:**
- **Erros não propagam:** cada nó captura sua exceção e a escreve em `state["errors"]`
  (lista acumulada via reducer `add`). Uma falha de rede no Financial não derruba a
  análise — os outros agentes seguem e a síntese trabalha com o que tem.
- **Structured outputs:** todo nó usa `with_structured_output` (Pydantic) — sem parse
  manual de texto do LLM.
- **Streaming agnóstico a HTTP:** o orquestrador emite `AnalysisEvent` (progresso por
  nó + evento final); a tradução para SSE mora só na camada da API.

---

## Stack

| Camada | Tecnologia |
|---|---|
| Orquestração | LangGraph |
| LLM | GPT-4o-mini (dev) / GPT-4o (prod) |
| Embeddings | text-embedding-3-small |
| Vector store | pgvector (PostgreSQL) |
| Cache | Redis |
| API | FastAPI + SSE (sse-starlette) |
| Web search | Tavily |
| Dados históricos | yfinance |
| Tracing | LangSmith |
| Métricas | Prometheus + Grafana |
| Deploy | Fly.io · Supabase (Postgres+pgvector) · Upstash (Redis) |

---

## Rodando localmente

### Pré-requisitos
- Python **3.12+**
- Docker + Docker Compose
- Chaves: `OPENAI_API_KEY`, `TAVILY_API_KEY` (LangSmith é opcional)

### Passo a passo

```bash
# 1. Configuração
cp .env.example .env
#    Edite o .env e preencha OPENAI_API_KEY e TAVILY_API_KEY (obrigatórias).
#    Em dev, deixe DATABASE_URL/REDIS_URL vazios — as partes POSTGRES_*/REDIS_*
#    já batem com o docker-compose.yml.

# 2. Infra local (Postgres+pgvector, Redis, Prometheus, Grafana)
docker compose up -d

# 3. Dependências (inclui ferramentas de dev: pytest, ruff, mypy)
pip install -e ".[dev]"

# 4. Migrations (cria as tabelas + extensão pgvector)
alembic upgrade head

# 5. Sobe a API
uvicorn finsight.api.main:app --reload
#    ou, via console script:  finsight-serve
```

A API fica em `http://localhost:8000`. Docs interativas (Swagger): `/docs`.

### Serviços do compose

| Serviço | Porta | Uso |
|---|---|---|
| PostgreSQL + pgvector | 5432 | vector store |
| Redis | 6379 | cache |
| Prometheus | 9090 | scrape de `/metrics` |
| Grafana | 3000 | dashboards (login anônimo em dev) |

---

## API

### `GET /health`
Liveness probe (usado pelo Fly.io). Retorna `{"status": "ok"}`.

### `GET /metrics`
Métricas no formato Prometheus. Retorna `404` se `PROMETHEUS_ENABLED=false`.

### `POST /analyze`
Dispara a análise multi-agente e faz **streaming via Server-Sent Events**: um evento
`progress` por nó concluído + um evento `complete` com a resposta final.

**Request:**
```json
{
  "query": "Vale a pena investir agora?",
  "ticker": "PETR4",
  "asset_type": "stock"
}
```
> `ticker` é normalizado para maiúsculo; `asset_type` aceita `stock` (default), `fii`, `etf`.

**Exemplo (curl):**
```bash
curl -N -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"query":"Vale a pena investir agora?","ticker":"PETR4"}'
```
A flag `-N` desabilita o buffer do curl para ver os eventos chegando em tempo real.

---

## Ingestão de relatórios (RAG)

O RAG Agent busca em documentos previamente indexados no pgvector. O pipeline de
ingestão (`src/finsight/ingestion/`) faz **chunk → embed → index**; a função pública é
`index_document(session, ticker=..., title=..., chunks=...)` (`ingestion/indexer.py`),
consumida após `chunker` + `embedder` gerarem os `EmbeddedChunk`. Sem documentos
indexados, o RAG Agent simplesmente retorna vazio — a análise segue com os outros dois.

---

## Deploy (Fly.io + Supabase + Upstash)

O Fly hospeda a **API** (container do `Dockerfile`); o **Postgres+pgvector** fica no
**Supabase** e o **Redis** na **Upstash**. Os segredos entram via `fly secrets` — nunca
no `fly.toml` nem na imagem.

```bash
# 0. Pré-requisito: provisione um projeto no Supabase (pgvector já vem habilitado)
#    e um banco Redis na Upstash. Anote as connection strings.

# 1. Cria o app no Fly sem publicar ainda
fly launch --no-deploy

# 2. Segredos (DATABASE_URL = Supabase, REDIS_URL = Upstash com rediss://)
#    DATABASE_URL: use o "Session pooler" (porta 5432), NÃO o "Transaction pooler"
#    (6543). O app usa asyncpg, que abre prepared statements nomeados; o pooler em
#    modo transação (6543) troca a conexão de servidor a cada transação e os prepared
#    statements "somem" → erros intermitentes. O Session pooler (5432) mantém a sessão
#    presa ao cliente e ainda é IPv4. Senha SÓ alfanumérica (caracteres como @ : / ?
#    quebram o parse da URL — o app não escapa a senha).
fly secrets set \
  OPENAI_API_KEY=sk-... \
  TAVILY_API_KEY=tvly-... \
  LANGCHAIN_API_KEY=ls-... \
  SECRET_KEY=$(openssl rand -hex 32) \
  DATABASE_URL='postgresql://postgres.<ref>:<senha>@aws-0-<region>.pooler.supabase.com:5432/postgres' \
  REDIS_URL='rediss://default:<token>@<endpoint>.upstash.io:6379'

# 3. Deploy — o release_command roda `alembic upgrade head` antes de subir o tráfego
fly deploy
```

**Como `DATABASE_URL`/`REDIS_URL` funcionam:** quando setados, **vencem** as partes
`POSTGRES_*`/`REDIS_*`. O app injeta o driver correto (`+asyncpg` no runtime,
`+psycopg2` no Alembic) e força **TLS** (`ssl=require` / `sslmode=require`) — exigência
do Supabase/Upstash. Em dev, ficam vazios e o app monta as URLs a partir das partes.

O `fly.toml` aplica as migrations no `release_command` (deploy aborta se falhar),
expõe health check em `/health` e usa **scale-to-zero** (a máquina hiberna sem tráfego).

---

## Variáveis de ambiente

Lista completa e comentada em [`.env.example`](.env.example). As essenciais:

| Variável | Obrigatória | Descrição |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | LLM + embeddings |
| `TAVILY_API_KEY` | ✅ | web search do Research Agent |
| `DATABASE_URL` | deploy | Postgres+pgvector (Supabase). Vazio em dev → usa `POSTGRES_*` |
| `REDIS_URL` | deploy | Redis (Upstash). Vazio em dev → usa `REDIS_*` |
| `LANGCHAIN_API_KEY` | opcional | tracing no LangSmith (no-op sem ela) |
| `ENVIRONMENT` | — | `development` / `production` (controla modelo e logging) |
| `PROMETHEUS_ENABLED` | — | expõe `/metrics` (default `true`) |

---

## Desenvolvimento

```bash
pytest tests/unit -q          # suíte unitária (sem rede — LLMs mockados)
ruff check src tests          # lint
mypy src                      # type-check (strict)
```

Os testes de integração (marcados `@pytest.mark.integration`) exigem `docker compose up`.
O CI (GitHub Actions) roda lint + mypy + unit a cada push.

### Estrutura

```
src/finsight/
├── api/            FastAPI: app (factory), main (entrypoint), routes (SSE), schemas
├── graph/          orchestrator (StateGraph diamante) + state (AgentState + modelos)
├── agents/         financial · research · rag · risk (um nó cada)
├── ingestion/      chunker · embedder · indexer (pipeline de PDF → pgvector)
├── retrieval/      retriever · hyde · reranker (RAG avançado)
├── evals/          dataset · generator · metrics · runner (avaliação do RAG)
├── observability/  metrics (Prometheus) · tracing (LangSmith)
└── db/             session (Settings + engine) · models · migrations (Alembic)
```

---

## Licença

Ver [LICENSE](LICENSE).
