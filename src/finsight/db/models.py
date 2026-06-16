"""
Modelos ORM SQLAlchemy — define o schema do banco como classes Python.

O Alembic detecta mudanças aqui via autogenerate e gera as migrations.
"""

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Index, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID, VARCHAR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from finsight.db.session import Base


class Document(Base):
    """
    Representa um documento financeiro indexado (relatório, RI, FII report).

    Um Document tem vários DocumentChunk — relação 1:N.
    Deletar o Document cascateia para os chunks (ON DELETE CASCADE na FK).
    """

    __tablename__ = "documents"

    # UUID gerado pelo banco (gen_random_uuid) — não pelo Python.
    # Motivo: consistência em bulk inserts e na migration — sem coordenação de IDs no app.
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    ticker: Mapped[str] = mapped_column(VARCHAR(10), nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)

    # source_type: categoriza o documento para filtros no RAG
    # ex: 'earnings' (resultados), 'ri' (relações com investidores), 'fii_report'
    source_type: Mapped[str | None] = mapped_column(VARCHAR(50), nullable=True)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    # TIMESTAMP WITH TIME ZONE: armazena timezone — sem isso, comparações entre
    # timezones diferentes retornam resultados incorretos
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relacionamento lazy="select" (default): chunks são carregados sob demanda.
    # Em produção, usamos queries explícitas — este relacionamento serve para
    # o cascade delete funcionar corretamente via ORM.
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="select",
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} ticker={self.ticker} title={self.title!r}>"


class DocumentChunk(Base):
    """
    Chunk de texto de um documento com seu embedding vetorial.

    Esta é a tabela central do RAG — cada linha é um trecho indexável.
    A busca por similaridade é feita via índice ivfflat nesta tabela.
    """

    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )

    # ON DELETE CASCADE: deletar o Document pai remove todos os chunks automaticamente
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Vector(1536): dimensão do text-embedding-3-small da OpenAI.
    # Mudar o modelo de embedding requer re-indexar todos os chunks
    # (dimensões incompatíveis causam erro no operador de similaridade).
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)

    # chunk_index: posição do chunk no documento original.
    # Útil para reconstruir contexto: se o chunk relevante é o índice 5,
    # podemos buscar os chunks 4 e 6 para dar mais contexto ao LLM.
    chunk_index: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # JSONB: metadados flexíveis por chunk (página do PDF, seção, etc.)
    # JSONB é indexável — diferente de JSON puro no Postgres
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata",  # nome da coluna no banco é "metadata" (sem underscore)
        JSONB,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    # Índice ivfflat para busca por similaridade vetorial.
    # Definido como __table_args__ porque envolve um operador customizado do pgvector
    # que o SQLAlchemy não sabe expressar via mapped_column.
    #
    # ivfflat vs hnsw:
    # - ivfflat: mais rápido para construir, usa menos memória, recall levemente menor
    # - hnsw: recall maior, build mais lento — preferível com >1M chunks
    # Para o volume deste projeto (dezenas de PDFs), ivfflat é suficiente.
    #
    # lists=100: número de clusters IVF. Regra geral: sqrt(n_rows).
    # Com <10k chunks, lists=100 já está acima do necessário mas não prejudica.
    __table_args__ = (
        Index(
            "ix_document_chunks_embedding_cosine",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"lists": 100},
        ),
    )

    def __repr__(self) -> str:
        return f"<DocumentChunk id={self.id} doc={self.document_id} idx={self.chunk_index}>"
