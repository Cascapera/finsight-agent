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
- [x] Semana 4: eval suite de RAG — dataset ✅, generator ✅, métricas ✅, runner ✅ (métricas próprias, não ragas-lib)
- [x] Semana 5: Orchestrator + Research + Financial + Risk Agent (StateGraph diamante)
- [x] Semana 6: RAG Agent (leque de 3 ramos) + streaming astream + API FastAPI/SSE
- [ ] Semana 7: Observabilidade (LangSmith + Prometheus)
- [ ] Semana 8: Deploy Fly.io + README final

**Semana atual:** 6 CONCLUÍDA — RAG Agent + API SSE. Os 3 passos ✅.
Próximo: Semana 7 (Observabilidade: LangSmith + Prometheus). PUSH pendente (main +4).
Limpeza Semana 4 feita: settings `ragas_*` → `eval_*` (`e9fbe88`); fix CI mypy/Python 3.12 (`45a3aed`).

**Semana 6 (commits `cbbf04b`/`c950718`/`2c07670`):** Passo 1 `agents/rag.py` — 3º ramo do leque,
casca fina sobre `retrieve_and_rerank`+`to_rag_output` (Sem.3), `_USE_HYDE=False`; orchestrator
virou LEQUE de 3 ramos. Passo 2 `run_analysis_stream` via `astream(stream_mode="updates")`,
emite `AnalysisEvent`(type/node/data) AGNÓSTICO a HTTP (progress por nó + complete c/ final_answer
+ erros acumulados). Passo 3 `api/` FastAPI: `schemas.AnalyzeRequest` (ticker→upper), `routes`
(GET /health, POST /analyze→EventSourceResponse traduz AnalysisEvent→evento SSE), `app.create_app`
(CORS + router; `uvicorn finsight.api.app:app`). 65 unit verdes. /metrics fica p/ Semana 7.

**Passo 3 Semana 5 — `agents/risk.py` + `graph/orchestrator.py` (commit `53a13c5`):** costura os
nós num grafo DIAMANTE executável. Risk Agent = nó de FAN-IN/síntese (4º modelo `FinalAnswer`):
`synthesize(state)→FinalAnswer` (lê financial/research/rag defensivamente, LLM structured output)
+ `risk_node` (captura erro→errors, fallback). DECISÃO: schema PRIVADO `_RiskVerdict` SEM
`disclaimer` — disclaimer é default FIXO do FinalAnswer (código), não do LLM. Orchestrator:
`StateGraph(AgentState)`, `add_edge(START,fin/research)`=FAN-OUT paralelo, `add_edge(fin/research,
risk)`=FAN-IN; reducer `add` acumula erros dos ramos. `build_orchestrator()` (compile sem
checkpointer), `build_initial_state()`, `run_analysis()` (o que a API SSE da Sem.6 chama).
Forward-compat RAG = +1 add_node +2 add_edge. 7 testes (inclui diamante e2e SEM rede). 57 unit.

**Passo 2 Semana 5 — `agents/research.py` (commit `d5e3077`):** complemento do Financial
(números vs "o que o mercado diz"). DIFERENÇA do Passo 1: DOIS pontos de rede. 3 camadas:
`_search_news(ticker,query)→list[NewsItem]` (Tavily SDK síncrona via `asyncio.to_thread`,
`topic="news"`; ponto de mock #1) + `analyze_sentiment(query,news)→ResearchOutput` (núcleo de
raciocínio, agnóstico à origem; LLM via `with_structured_output`, `_get_chat_client` temp=0.0 =
ponto de mock #2) + `research_node(state)→dict` (os 2 IO no mesmo try, captura→`errors`, nunca
propaga). DECISÃO-CHAVE: schema PRIVADO `_SentimentVerdict` SEM `sources` — o LLM julga, mas as
URLs reais vêm dos NewsItem (código), não do modelo. Guard sem notícias→output neutro SEM LLM.
8 testes. 50 unit verdes.

**Passo 1 Semana 5 — `agents/financial.py` (commit local, pré-push):** primeiro nó do grafo,
único DETERMINÍSTICO (sem LLM). 2 camadas: `compute_metrics(prices)→FinancialOutput` (núcleo
quant puro: Sharpe anualizado, VaR 95% histórico como perda positiva, retorno acum. 1a, vol
anualizada; testável sem rede) + `financial_node(state)→dict[str,Any]` (contrato de nó LangGraph:
busca via yfinance em `asyncio.to_thread`, captura erro em `state["errors"]`, nunca propaga).
`_fetch_prices` = ponto de rede/mock. pandas.* no ignore do mypy. 9 testes. 42 unit verdes.

**Decisão Semana 4 (2026-06-18):** RAGAS-a-biblioteca NÃO importa na stack
langchain v1 (`langchain_community 0.4.2` removeu `chat_models.vertexai`, que TODA
versão do ragas até a 0.4.3 importa incondicionalmente). Guilherme escolheu
**construir as métricas nós mesmos** (LLM-as-judge, mesmo padrão do reranker) —
mockáveis, CI sem rede, zero dep frágil. RAGAS fica como referência conceitual.
LIMPEZA FEITA (Passo 4): `ragas` removido do pyproject (deps + `ignore_missing_imports`).
Ainda instalado no `.venv` (não atrapalha); some num `pip install -e` em ambiente novo.

**Onde paramos (2026-06-24) — Semana 4 CONCLUÍDA:**
- ✅ Passo 1 — `evals/dataset.py`: `EvalSample`/`EvalDataset` (Pydantic strict),
  golden set. Mapa métrica→campo no topo do arquivo. `from_json`/`to_json`,
  `filter_by_ticker`, `SEED_DATASET` (3 casos fictícios "Petro Norte"). Commit `154276b`.
- ✅ Passo 2 — `evals/generator.py`: o "G" do RAG. `generate_answer(question, contexts)`
  ancora SÓ no contexto (recusa se faltar info — é o que faithfulness mede). Agnóstico à
  origem dos chunks, temperature=0.0, client mockável. Guard de contexto vazio → recusa
  determinística sem LLM. Commit `0e2a0e7`.
- ✅ Passo 3 — `evals/metrics.py`: 4 métricas LLM-as-judge. Sacada: 4 métricas = 2 padrões.
  Padrão A (`_classify_claims`, decompõe texto → claims → suporte): faithfulness (decompõe
  resposta vs contextos) + context_recall (decompõe ground_truth vs contextos). Padrão B:
  context_precision (juiz rotula chunks → Average Precision determinística premia relevante no
  topo) + answer_relevancy (juiz gera perguntas-reversas → cosseno médio via `embed_texts` +
  flag noncommittal zera evasão). `_get_judge_client` lru_cache temp=0.0 = ponto de mock (client
  cru, cada métrica faz `.with_structured_output(schema)`). Cada métrica → `MetricResult(score,
  details)`. Guards sem LLM. 20 testes. Commit `ff33393`.
- ✅ Passo 4 — `evals/runner.py`: COSTURA tudo. Estratégia de retrieval como Protocol injetável
  (`StrategyFn` + wrappers `_strategy_baseline/_hyde/_rerank`, assinatura única) → runner é
  orquestração PURA, testável sem banco. `evaluate_sample` (recupera→gera→mede 4 métricas em
  `asyncio.gather`), `run_evaluation` (estratégia×sample, sequencial p/ poupar rate limit),
  `aggregate`+`format_report` (puras → tabela markdown comparativa). 5 testes. ragas removido
  do pyproject. 33 testes unitários verdes. Commit `<pendente>`.

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
  - **SEM linha `Co-Authored-By`** em commits e PRs (pedido do Guilherme, 2026-06-25)

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
