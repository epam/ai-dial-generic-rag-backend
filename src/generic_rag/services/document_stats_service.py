from collections.abc import Sequence
from typing import Annotated

from injection import scoped
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from generic_rag.channel import Channel
from generic_rag.db.entities import DocumentEntity, ImageChunkEntity, TextChunkEntity
from generic_rag.db.session import get_current_session, transaction
from generic_rag.scope import ScopeName


class DocumentStats(BaseModel):
    """ Document statistics. """
    document_id: Annotated[int, Field(..., description="id of the document")]
    number_of_pages: Annotated[int | None, Field(description="Number of document pages", ge=0)] = None


@scoped(ScopeName.channel)
class DocumentStatsService:
    """ Service for getting documents statistics. """

    def __init__(self, channel: Channel):
        self._channel_key = channel.channel_key

    @transaction
    async def get_document_stats(self, *document_ids: int) -> Sequence[DocumentStats]:
        """ Get statistics for given documents. """
        if not document_ids:
            return []

        result = await get_current_session().execute(
            select(
                DocumentEntity.document_id,
                func.max(
                    func.greatest(
                        TextChunkEntity.page_number,
                        ImageChunkEntity.page_number,
                    )
                ).label(
                    "number_of_pages"
                )
            ).outerjoin(
                TextChunkEntity
            ).outerjoin(
                ImageChunkEntity
            ).where(
                DocumentEntity.channel_key == self._channel_key,
                DocumentEntity.document_id.in_(
                    set(document_ids)
                ),
            ).group_by(
                DocumentEntity.document_id
            )
        )
        return [
            DocumentStats(
                document_id=id_,
                number_of_pages=number_of_pages,
            ) for id_, number_of_pages in result.all()
        ]
