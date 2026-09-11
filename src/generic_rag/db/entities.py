import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKeyConstraint,
    PrimaryKeyConstraint,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from generic_rag.types import DocumentStatus, ImageType


class _EntityBase(DeclarativeBase): ...


class DocumentEntity(_EntityBase):
    __tablename__ = "documents"

    channel_key: Mapped[str]
    document_id: Mapped[int]

    url: Mapped[str]
    etag: Mapped[str | None]
    mime_type: Mapped[str]
    size: Mapped[int]

    display_name: Mapped[str]
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB)

    status: Mapped[DocumentStatus] = mapped_column(Enum(DocumentStatus, native_enum=False, length=15))

    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (PrimaryKeyConstraint("channel_key", "document_id"),)


class TextChunkEntity(_EntityBase):
    __tablename__ = "text_chunks"

    channel_key: Mapped[str]
    document_id: Mapped[int]
    chunk_id: Mapped[int]

    text: Mapped[str] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB)

    __table_args__ = (
        PrimaryKeyConstraint("channel_key", "document_id", "chunk_id"),
        ForeignKeyConstraint(
            ["channel_key", "document_id"],
            [DocumentEntity.channel_key, DocumentEntity.document_id],
            ondelete="cascade",
        ),
    )


class ImageChunkEntity(_EntityBase):
    __tablename__ = "image_chunks"

    channel_key: Mapped[str]
    document_id: Mapped[int]
    chunk_id: Mapped[int]

    image_type: Mapped[ImageType] = mapped_column(Enum(ImageType, native_enum=False, length=15))
    image_url: Mapped[str]
    mime_type: Mapped[str]
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB)

    __table_args__ = (
        PrimaryKeyConstraint("channel_key", "document_id", "chunk_id"),
        ForeignKeyConstraint(
            ["channel_key", "document_id"],
            [DocumentEntity.channel_key, DocumentEntity.document_id],
            ondelete="cascade",
        ),
    )
