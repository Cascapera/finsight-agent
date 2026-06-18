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
- [x] Semana 2: infra base + pgvector + ingestão de PDFs + CI (GitHub Actions)
- [x] Semana 3: RAG avançado — retriever base ✅, HyDE ✅, re-ranking ✅
- [~] Semana 4: eval suite de RAG — dataset ✅, generator ✅, métricas ⬜, runner ⬜ (métricas próprias, não ragas-lib)
- [ ] Semana 5: Orchestrator + Research + Financial Agent
- [ ] Semana 6: RAG Agent + API SSE completa
- [ ] Semana 7: Observabilidade (LangSmith + Prometheus)
- [ ] Semana 8: Deploy Fly.io + README final

**Semana atual:** 4 EM ANDAMENTO (Passos 1-2 feitos) — eval suite de RAG

**Decisão Semana 4 (2026-06-18):** RAGAS-a-biblioteca NÃO importa na stack
langchain v1 (`langchain_community 0.4.2` removeu `chat_models.vertexai`, que TODA
versão do ragas até a 0.4.3 importa incondicionalmente). Guilherme escolheu
**construir as métricas nós mesmos** (LLM-as-judge, mesmo padrão do reranker) —
mockáveis, CI sem rede, zero dep frágil. RAGAS fica como referência conceitual.
TODO de limpeza: remover `ragas` do pyproject no fim da semana.

**Onde paramos (2026-06-18) — Semana 4:**
- ✅ Passo 1 — `evals/dataset.py`: `EvalSample`/`EvalDataset` (Pydantic strict),
  golden set. Mapa métrica→campo no topo do arquivo. `from_json`/`to_json`,
  `filter_by_ticker`, `SEED_DATASET` (3 casos fictícios "Petro Norte"). Commit `154276b`.
- ✅ Passo 2 — `evals/generator.py`: o "G" do RAG. `generate_answer(question, contexts)`
  ancora SÓ no contexto (recusa se faltar info — é o que faithfulness mede). Agnóstico à
  origem dos chunks, temperature=0.0, client mockável. Guard de contexto vazio → recusa
  determinística sem LLM. Commit `0e2a0e7`.
- ⬜ **PRÓXIMO: Passo 3 — `evals/metrics.py`**: 4 métricas LLM-as-judge (faithfulness,
  answer_relevancy, context_precision, context_recall). Depois Passo 4 = `runner.py`
  (roda baseline/HyDE/rerank sobre o golden set → tabela comparativa).

Semana 3 (CONCLUÍDA): retriever base (`61e9fd2`), HyDE (`5df6680`), re-ranking (`84fa275`).
26 testes verdes (18 da Semana 3 + 8 novos: dataset + generator). Detalhes: memória `project_state.md`.

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
