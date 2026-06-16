"""create documents tables

Revision ID: 0001
Revises:
Create Date: 2026-05-25 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None  # primeira migration — sem predecessor
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Habilita a extensão pgvector — idempotente com IF NOT EXISTS
    # Precisa rodar antes de criar colunas do tipo VECTOR
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Habilita gen_random_uuid() — disponível via pgcrypto
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    # ─── Tabela documents ───────────────────────────────────────────────────
    op.create_table(
        "documents",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("ticker", sa.VARCHAR(10), nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("source_type", sa.VARCHAR(50), nullable=True),
        sa.Column("source_url", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_documents_ticker", "documents", ["ticker"])

    # ─── Tabela document_chunks ─────────────────────────────────────────────
    op.create_table(
        "document_chunks",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        # VECTOR(1536): dimensão do text-embedding-3-small
        # SQL raw porque o tipo vector não existe no sqlalchemy core
        sa.Column(
            "embedding",
            sa.Text,  # placeholder — substituído abaixo via ALTER TABLE
            nullable=True,
        ),
        sa.Column("chunk_index", sa.Integer, nullable=True),
        sa.Column("metadata", postgresql.JSONB, nullable=True),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            ondelete="CASCADE",  # deleta chunks quando o documento pai é deletado
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_document_chunks_document_id", "document_chunks", ["document_id"])

    # Altera a coluna embedding para o tipo VECTOR(1536) real
    # Não podemos declarar VECTOR diretamente no create_table acima porque
    # o tipo não existe no dialeto padrão do SQLAlchemy — usamos SQL raw
    op.execute("ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector(1536) USING NULL")

    # ─── Índice ivfflat para busca por similaridade ─────────────────────────
    # ivfflat com vector_cosine_ops: otimizado para busca por similaridade de cosseno
    # lists=100: número de clusters — regra geral sqrt(n_rows esperado)
    # IMPORTANT: este índice só pode ser criado depois que a tabela tem dados
    # (o Postgres precisa de pelo menos 1 linha para calibrar os clusters IVF)
    # Em produção, crie o índice após a primeira ingestão de documentos.
    # Para dev/CI, criamos aqui mesmo para que a estrutura esteja completa.
    op.execute(
        """
        CREATE INDEX ix_document_chunks_embedding_cosine
        ON document_chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        """
    )


def downgrade() -> None:
    op.drop_table("document_chunks")
    op.drop_table("documents")
    # Não removemos as extensões vector/pgcrypto no downgrade —
    # outras partes do banco podem depender delas
