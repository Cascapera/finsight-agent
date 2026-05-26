"""
Chunking de documentos PDF para ingestão no pipeline RAG.

Responsabilidade: receber um PDF em bytes, extrair texto por página,
dividir em chunks respeitando limites de token, retornar chunks com metadados.
"""

import io
from dataclasses import dataclass, field

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from finsight.db.session import settings


@dataclass
class ChunkResult:
    """
    Resultado do chunking de um documento.

    Separamos texto e metadados porque o embedder precisa do texto puro
    e o indexer precisa dos metadados para salvar junto com o embedding.
    """

    content: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _build_length_function(encoding_name: str = "cl100k_base"):
    """
    Retorna uma função que conta tokens usando tiktoken.

    cl100k_base: encoding usado pelo text-embedding-3-small e GPT-4.
    Passamos esta função ao RecursiveCharacterTextSplitter para que
    chunk_size seja interpretado em tokens, não em caracteres.

    Por que factory e não lambda direta?
    tiktoken.get_encoding() é uma chamada relativamente cara (lê arquivo de vocab).
    Fazemos uma vez e capturamos o encoder no closure — sem re-inicialização
    a cada chamada de len().
    """
    enc = tiktoken.get_encoding(encoding_name)

    def count_tokens(text: str) -> int:
        return len(enc.encode(text))

    return count_tokens


def build_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """
    Constrói o splitter com os parâmetros das Settings (ou valores customizados).

    Separamos a construção do splitter da função de chunking para facilitar
    testes unitários — nos testes podemos injetar um splitter com chunk_size=10
    sem depender das Settings.

    Args:
        chunk_size: tamanho máximo do chunk em tokens. Default: settings.chunk_size_tokens
        chunk_overlap: overlap entre chunks em tokens. Default: settings.chunk_overlap_tokens
    """
    size = chunk_size or settings.chunk_size_tokens
    overlap = chunk_overlap or settings.chunk_overlap_tokens

    return RecursiveCharacterTextSplitter(
        # separators em ordem de preferência:
        # 1. parágrafo duplo (maior unidade semântica)
        # 2. nova linha simples
        # 3. ponto final (fim de sentença)
        # 4. espaço (última opção — quebra palavras só se necessário)
        # 5. "" (caractere por caractere — nunca deve chegar aqui com textos normais)
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=_build_length_function(),
        # add_start_index=True: adiciona o índice de caractere de início do chunk
        # nos metadados — útil para debug e para localizar o trecho no PDF original
        add_start_index=True,
        # is_separator_regex=False: os separators são strings literais, não regex
        is_separator_regex=False,
    )


def extract_text_from_pdf(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """
    Extrai texto de um PDF em bytes, página por página.

    Returns:
        Lista de (page_number, text) — page_number começa em 1 (convenção humana).

    Por que retornar por página e não texto concatenado?
    Manter a granularidade de página nos metadados do chunk permite que o RAG
    cite a fonte com precisão: "conforme página 12 do Relatório de Resultados Q3".
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages: list[tuple[int, str]] = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        # Normaliza espaços em branco excessivos — PDFs financeiros costumam ter
        # artefatos de formatação (tabulações, espaços múltiplos entre colunas)
        text = " ".join(text.split())
        if text.strip():  # ignora páginas em branco ou só com imagens
            pages.append((page_num, text))

    return pages


def chunk_document(
    pdf_bytes: bytes,
    ticker: str,
    title: str,
    source_type: str | None = None,
    source_url: str | None = None,
    splitter: RecursiveCharacterTextSplitter | None = None,
) -> list[ChunkResult]:
    """
    Pipeline completo: PDF → chunks com metadados.

    Args:
        pdf_bytes: conteúdo do arquivo PDF em bytes
        ticker: código do ativo (ex: "PETR4") — para filtros no RAG
        title: título do documento — aparece nas citações
        source_type: categoria do documento ("earnings", "ri", "fii_report")
        source_url: URL de origem, se disponível
        splitter: instância customizada do splitter (útil em testes)

    Returns:
        Lista de ChunkResult ordenada por (página, chunk_index_na_página)
    """
    _splitter = splitter or build_splitter()
    pages = extract_text_from_pdf(pdf_bytes)

    results: list[ChunkResult] = []
    global_chunk_index = 0

    for page_num, page_text in pages:
        # Divide o texto desta página em chunks
        # split_text retorna list[str] — os metadados de start_index ficam
        # disponíveis via create_documents (retorna Document com metadata)
        page_chunks = _splitter.split_text(page_text)

        for local_index, chunk_text in enumerate(page_chunks):
            results.append(
                ChunkResult(
                    content=chunk_text,
                    chunk_index=global_chunk_index,
                    metadata={
                        "ticker": ticker,
                        "title": title,
                        "source_type": source_type,
                        "source_url": source_url,
                        "page_number": page_num,
                        "chunk_index_in_page": local_index,
                        "total_pages": len(pages),
                    },
                )
            )
            global_chunk_index += 1

    return results
