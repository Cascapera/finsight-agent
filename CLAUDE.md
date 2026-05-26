# FinSight Agent — Contexto do Projeto

## Quem é o Guilherme

Backend Engineer em transição para AI/LLM Engineer. Background sólido em:
- Python produção: Django, FastAPI, Celery, Redis, Docker, AWS
- Infra: Prometheus, Grafana, filas, idempotência, retry com backoff
- Fintech: cálculos de Sharpe, VaR, retorno acumulado

Está aprendendo agora: LangGraph, pgvector, RAG em produção, multi-agent systems, LangSmith, RAGAS.

## Papel do Claude Code neste projeto

**Professor técnico + engenheiro par.**

Antes de criar qualquer arquivo:
1. Explique **o que** vai ser criado e **por que** esta abordagem
2. Escreva o código com **comentários técnicos densos** nas decisões não-óbvias
3. Ao final de cada passo, pergunte **"Posso continuar?"** e aguarde

Não teste o entendimento do Guilherme com perguntas. Se ele quiser tirar dúvida, ele pergunta diretamente.

## O que está construído (atualizar a cada semana)

- [x] Estrutura de pastas do projeto
- [x] `src/finsight/graph/state.py` — AgentState + modelos Pydantic
- [ ] Semana 2: infra base + pgvector + ingestão de PDFs
- [ ] Semana 3: RAG avançado (HyDE, re-ranking)
- [ ] Semana 4: RAGAS eval suite
- [ ] Semana 5: Orchestrator + Research + Financial Agent
- [ ] Semana 6: RAG Agent + API SSE completa
- [ ] Semana 7: Observabilidade (LangSmith + Prometheus)
- [ ] Semana 8: Deploy Fly.io + README final

**Semana atual:** 2 — infra base + pgvector
**Próximo passo:** `pyproject.toml`

## Arquitetura

```
FastAPI (SSE) → Orchestrator (LangGraph)
                  ├── Research Agent  → Tavily → notícias + sentimento
                  ├── Financial Agent → yfinance → Sharpe, VaR, retorno
                  └── RAG Agent       → pgvector → chunks de relatórios PDF
                → Risk Agent (síntese final)
                → Redis cache (TTL 1h)
                → LangSmith (tracing) + Prometheus (métricas)
```

## AgentState — não alterar sem alinhamento

```python
class AgentState(TypedDict):
    query: str
    ticker: str
    asset_type: str
    research:     ResearchOutput | None
    financial:    FinancialOutput | None
    rag:          RAGOutput | None
    final_answer: FinalAnswer | None
    execution_id: str
    cached: bool
    errors: Annotated[list[str], add]   # acumulado por todos os nós
```

## Stack

| Camada | Tecnologia |
|---|---|
| Orquestração | LangGraph |
| LLM | GPT-4o-mini (dev) / GPT-4o (prod) |
| Embeddings | text-embedding-3-small |
| Vector store | pgvector (PostgreSQL) |
| Cache | Redis |
| API | FastAPI + SSE |
| Web search | Tavily |
| Dados históricos | yfinance |
| Tracing | LangSmith |
| Métricas | Prometheus + Grafana |
| Evals | RAGAS |
| Deploy | Fly.io |

## Convenções de código

- **Type hints** em tudo — sem `Any` desnecessário
- **Pydantic v2** para todos os modelos de dados
- **Async** nas funções de IO (banco, Redis, chamadas HTTP)
- **Structured outputs** em todos os nós LangGraph (sem parse manual de texto)
- **Erros:** capturar no nó, escrever em `state["errors"]`, nunca propagar exceção
- **Testes:** cada agente tem mock de LLM — CI não faz chamadas reais ao OpenAI
- **Commits:** mensagens em inglês, padrão Conventional Commits
  - `feat:`, `fix:`, `chore:`, `test:`, `docs:`, `refactor:`

## Permissões pré-aprovadas

Definidas em `.claude/settings.json`. Resumo:
- ✅ `git add/commit/push` — liberados
- ✅ Bash: pip, python, pytest, ruff, mypy, uvicorn
- ✅ Docker Compose: up/down/logs/ps/build
- ✅ Alembic, psql, redis-cli
- ✅ Operações de arquivo: mkdir, touch, cp, mv, cat, ls
- ❌ `git push --force`, `git reset --hard`, `git rebase`
- ❌ PR — Guilherme autoriza manualmente via `gh pr create`
- ❌ `rm -rf` e qualquer deleção de arquivo/pasta
- ❌ `docker system prune`, `DROP TABLE`, `TRUNCATE`

## Referências do projeto

- Repo: `git@github.com:Cascapera/finsight-agent.git`
- SDD completo: `docs/SDD.md`
- Plano de estudos: 16 semanas, 4 meses
