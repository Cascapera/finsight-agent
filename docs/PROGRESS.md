# FinSight Agent — Progresso & Runbook

> Registro durável do estado do projeto, do deploy e dos próximos passos.
> Versionado no repo de propósito — não depende do sistema de memória do assistente.

## Estado atual (2026-06-29)

**PROJETO COMPLETO (16 semanas) + DEPLOY REAL FEITO e VERIFICADO end-to-end.**

- 🌐 **No ar:** https://finsight-agent.fly.dev/
- ✅ Os **4 ramos** funcionando em produção: Financial + Research + RAG + Risk
- ✅ `HEAD == origin/main` (`78a7e1f`), working tree limpo, 91 testes unitários verdes
- **Infra:** Fly.io (app, região `gru`, 2 máquinas HA scale-to-zero) · Supabase (Postgres+pgvector,
  region us-east-1) · Upstash (Redis TLS)

### Verificação E2E (`POST /analyze`, ticker PETR4)
- **Financial** → Sharpe 1.27 · VaR 2.5% · retorno 1a +32.5% · preço R$ 38,23 (via `PETR4.SA`)
- **Research** → notícias reais Petrobras (investimentos, bloco de Campos) · sentimento bullish · conf 0.8
- **RAG** → 5 chunks do relatório 1T26 ingerido (EBITDA R$ 61,7 bi, lucro R$ 23,8 bi, FCO R$ 44 bi)
- **Risk** → síntese cruzando os 3 · `errors: []`

---

## Runbook de deploy (Fly + Supabase + Upstash)

> `fly` não está no PATH da sessão Bash do assistente: `export PATH="$PATH:$HOME/.fly/bin"`.
> No PowerShell do Guilherme: `$env:Path += ";$HOME\.fly\bin"`.

1. **App:** `fly apps create finsight-agent` (já feito).
2. **Secrets** (PowerShell — aspas simples, crase ` como continuador):
   ```powershell
   fly secrets set `
     OPENAI_API_KEY='sk-...' `
     TAVILY_API_KEY='tvly-...' `
     SECRET_KEY='<python -c "import secrets; print(secrets.token_hex(32))">' `
     DATABASE_URL='postgresql://postgres.<ref>:<senha>@aws-1-us-east-1.pooler.supabase.com:5432/postgres' `
     REDIS_URL='rediss://default:<token>@<endpoint>.upstash.io:6379'
   ```
3. **Deploy:** `fly deploy` (o `release_command` roda `alembic upgrade head` no Supabase antes de rotear).
4. **Verificar:** `curl https://finsight-agent.fly.dev/health` → `{"status":"ok"}`.

### Ingestão de PDF (popular o RAG)
Script: `scripts/ingest_pdf.py`. Roda **local** apontando para o Supabase via env vars:
```powershell
$env:DATABASE_URL='postgresql://postgres.<ref>:<senha>@aws-1-us-east-1.pooler.supabase.com:5432/postgres'
# OPENAI_API_KEY vem do .env
python scripts/ingest_pdf.py "C:\caminho\relatorio.pdf" --ticker PETR4 --title "Petrobras 1T26" --source-type earnings
```
Teste de conexão barato (sem gastar OpenAI): `python -m alembic current` → deve imprimir `0001 (head)`.

---

## Fixes / pegadinhas do deploy real (já resolvidos)

| # | Problema | Causa | Fix |
|---|---|---|---|
| 1 | `release_command` rodava `uvicorn ... alembic upgrade head` | Fly substitui CMD mas ANEXA a um ENTRYPOINT | Dockerfile: comando inteiro em `CMD`, sem `ENTRYPOINT` (`709d123`) |
| 2 | asyncpg falhava intermitente / lento | Supabase **transaction** pooler (6543) quebra prepared statements | usar **session** pooler (5432) (`e65e04e`) |
| 3 | `password authentication failed` | senha do Postgres com caractere especial quebra o parse da URL | senha **só alfanumérica** |
| 4 | 401 OpenAI em todos os nós LLM | secret `OPENAI_API_KEY` no Fly tinha valor inválido (≠ do .env) | re-set do secret (`.env` e `fly secrets` são INDEPENDENTES; Fly NÃO lê .env) |
| 5 | Financial vinha vazio | `PETR4` no Yahoo retorna "not found"; B3 precisa de `.SA` | helper `_to_yahoo_symbol` (`e425e57`) |
| 6 | Research trazia notícias off-topic | query `{ticker} {query}` crua | `_build_news_query` ancora em termos financeiros + `days=30` (`78a7e1f`) |

**Lição permanente:** antes de `git push`, rodar `.venv/Scripts/python.exe -m ruff format --check .`
no **repo inteiro** (CI roda assim; só `ruff check` não pega formatação).

---

## Próximo passo (escolhido 2026-06-29, NÃO iniciado): "Demo + Observabilidade"

Estado: só `docker/prometheus.yml` + serviços prometheus/grafana no `docker-compose.yml`. Grafana sobe
VAZIO; fly.toml sem `[metrics]`. Métricas existentes: `finsight_node_{duration_seconds,runs_total,
errors_total}` (label `node`), expostas em `GET /metrics`.

### A) Grafana provisionado (dev local) — acordado, eu faço
- `docker/grafana/provisioning/datasources/prometheus.yml` → datasource `http://prometheus:9090`
- `docker/grafana/provisioning/dashboards/dashboards.yml` → provider carregando dashboards de pasta
- `docker/grafana/dashboards/finsight.json` → 4 painéis:
  1. Throughput por nó: `sum by (node) (rate(finsight_node_runs_total[5m]))`
  2. Taxa de erro por nó: `sum by (node) (rate(finsight_node_errors_total[5m])) / sum by (node) (rate(finsight_node_runs_total[5m]))`
  3. Latência p95: `histogram_quantile(0.95, sum by (node, le) (rate(finsight_node_duration_seconds_bucket[5m])))`
  4. Latência p50: idem com `0.5`
- Montar essas pastas no serviço `grafana` do `docker-compose.yml` (hoje só monta `grafanadata`).

### B) Métricas em PROD (Fly) — acordado, eu faço
- Adicionar ao `fly.toml`:
  ```toml
  [metrics]
    port = 8000
    path = "/metrics"
  ```
- `fly deploy` → Prometheus gerenciado do Fly scrapeia → métricas no Grafana do Fly (`fly-metrics.net`).

### C) LangSmith em prod — OPCIONAL, precisa da chave do Guilherme
- `fly.toml` já liga `LANGCHAIN_TRACING_V2=true`; falta o secret.
- `fly secrets set LANGCHAIN_API_KEY='ls-...'` → traces de cada `/analyze` no LangSmith.

---

## Evoluções futuras mapeadas (não priorizadas)

- **Dedup via `content_hash`** — re-ingerir o mesmo PDF hoje DUPLICA chunks (`indexer.py` admite o TODO).
  Migration `0002` + checagem no indexer. É o gap mais relevante de produção.
- **Ingestão como task Celery** — endpoint enfileira → worker processa com retry/backoff. Encaixa no
  forte do Guilherme (filas/idempotência) e é o melhor capítulo "produção" do portfólio.
- **Verificar cache Redis do `/analyze`** — arquitetura promete TTL 1h e `AgentState.cached` existe;
  confirmar se está realmente implementado (orchestrator compila SEM checkpointer).

## Detalhes cosméticos conhecidos (não-bloqueantes)
- Citações do RAG mostram "Petrobras 4T25" (título digitado errado na ingestão — é 1T26). Corrigir
  exige re-ingerir, que duplicaria chunks enquanto não houver dedup. Guilherme: "para teste tá ok".
