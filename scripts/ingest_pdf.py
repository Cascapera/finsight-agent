"""
Script de ingestão de PDFs para o RAG (pgvector).

Costura o pipeline de ingestão (chunker -> embedder -> indexer) numa CLI pontual.
NÃO é um endpoint: ingestão é uma operação batch, e o `index_document` foi desenhado
para o chamador controlar a transação (ele faz flush, não commit). Aqui o script é
o "chamador" que abre a sessão, ingere e dá o commit atômico.

Onde roda: localmente, mas apontando para o banco que você quiser via env vars. Para
gravar no Supabase de produção, defina DATABASE_URL (session pooler, porta 5432) e
OPENAI_API_KEY (real — embeddings consomem tokens) ANTES de rodar. O Settings lê de
os.environ, que vence o .env, então o mesmo script serve para dev local e produção
sem editar arquivo nenhum.

Uso (PowerShell):
    $env:DATABASE_URL='postgresql://postgres.<ref>:<senha>@aws-1-us-east-1.pooler.supabase.com:5432/postgres'
    $env:OPENAI_API_KEY='sk-...'
    python scripts/ingest_pdf.py relatorio.pdf --ticker PETR4 --title "Petrobras 4T25"

    # vários PDFs do mesmo ativo de uma vez (o --title default vira o nome do arquivo):
    python scripts/ingest_pdf.py docs/*.pdf --ticker PETR4
"""

import argparse
import asyncio
import logging
from pathlib import Path

# Import do pacote instalado (editable no venv). PYTHONPATH não é necessário aqui.
from finsight.db.session import AsyncSessionLocal
from finsight.ingestion.chunker import chunk_document
from finsight.ingestion.embedder import embed_chunks
from finsight.ingestion.indexer import index_document

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ingest_pdf")


async def ingest_one(
    pdf_path: Path,
    *,
    ticker: str,
    title: str,
    source_type: str | None,
    source_url: str | None,
) -> None:
    """Ingere um único PDF: chunk -> embed -> index -> commit."""
    pdf_bytes = pdf_path.read_bytes()

    # 1. PDF -> chunks com metadados de página (síncrono, CPU-bound leve).
    chunks = chunk_document(
        pdf_bytes,
        ticker=ticker,
        title=title,
        source_type=source_type,
        source_url=source_url,
    )
    if not chunks:
        # PDF só com imagens/escaneado sem OCR -> extract_text não acha texto.
        logger.warning("Nenhum texto extraído de %s — pulando.", pdf_path.name)
        return

    # 2. chunks -> embeddings (rede: OpenAI; batelado dentro de embed_chunks).
    embedded = await embed_chunks(chunks)

    # 3. grava Document + DocumentChunks numa transação atômica.
    # AsyncSessionLocal vem de db.session, ligado ao engine montado do Settings
    # (DATABASE_URL aponta para onde os env vars mandarem).
    async with AsyncSessionLocal() as session:
        doc = await index_document(
            session,
            ticker=ticker,
            title=title,
            chunks=embedded,
            source_type=source_type,
            source_url=source_url,
        )
        await session.commit()  # o script é o dono da transação

    logger.info("OK: %s -> doc=%s (%d chunks)", pdf_path.name, doc.id, len(embedded))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingere PDFs no RAG (pgvector).")
    parser.add_argument("pdfs", nargs="+", type=Path, help="Caminho(s) do(s) PDF(s).")
    parser.add_argument("--ticker", required=True, help='Código do ativo (ex: "PETR4").')
    parser.add_argument(
        "--title",
        default=None,
        help="Título do documento (default: nome do arquivo). Aparece nas citações.",
    )
    parser.add_argument(
        "--source-type",
        default=None,
        help='Categoria: "earnings", "ri", "fii_report", etc.',
    )
    parser.add_argument("--source-url", default=None, help="URL de origem, se houver.")
    args = parser.parse_args()

    # ticker normalizado em maiúsculas — bate com o asset_type/ticker do RAG Agent.
    ticker = args.ticker.upper()

    for pdf_path in args.pdfs:
        if not pdf_path.is_file():
            logger.error("Arquivo não encontrado: %s — pulando.", pdf_path)
            continue
        # title default = nome do arquivo sem extensão, quando ingerindo em lote.
        title = args.title or pdf_path.stem
        await ingest_one(
            pdf_path,
            ticker=ticker,
            title=title,
            source_type=args.source_type,
            source_url=args.source_url,
        )


if __name__ == "__main__":
    asyncio.run(main())
