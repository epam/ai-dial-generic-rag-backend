"""The citation text `PlainAnswer` writes, which is what an MCP `rag_search` caller reads."""

from generic_rag.types import ChunkMetadata, RetrievedDocument, TextChunk
from generic_rag.utils.answers import PlainAnswer


def _document(*, document_id: int = 207, page_number: int = 1) -> RetrievedDocument:
    return RetrievedDocument(
        chunks=[
            TextChunk(
                document_id=document_id,
                chunk_id=1,
                text="CHUNK_BODY",
                metadata=ChunkMetadata(page_number=page_number),
            )
        ],
        source_id=document_id,
        source_url="files/BUCKET-ID/appdata/deployment/reports/Some%20Report%202024.pdf",
        source_page_number=page_number,
        source_display_name="Some Report 2024.pdf",
    )


async def test_citation_names_the_document_and_the_page():
    answer = PlainAnswer()

    await answer.add_citation(1, _document(document_id=207, page_number=1))

    assert answer.content == "[Document 207, Page 1]"


async def test_citation_matches_the_form_get_pages_emits():
    """One server, one citation form: `get_pages` labels its pages the same way."""
    document_id, page_number = 42, 7
    answer = PlainAnswer()

    await answer.add_citation(1, _document(document_id=document_id, page_number=page_number))

    assert answer.content == f"[Document {document_id}, Page {page_number}]"


async def test_citation_appends_to_the_surrounding_text():
    answer = PlainAnswer()

    answer.append_content("The deficit grew in 2024.")
    await answer.add_citation(1, _document(document_id=3, page_number=12))

    assert answer.content == "The deficit grew in 2024.[Document 3, Page 12]"


async def test_a_citation_marks_the_answer_as_having_references():
    answer = PlainAnswer()
    assert not answer.has_references

    await answer.add_citation(1, _document())

    assert answer.has_references


async def test_each_citation_is_written_out_in_full():
    """Two citations of one document stay independently readable, with no shared prefix."""
    answer = PlainAnswer()

    await answer.add_citation(1, _document(document_id=9, page_number=2))
    await answer.add_citation(1, _document(document_id=9, page_number=5))

    assert answer.content == "[Document 9, Page 2][Document 9, Page 5]"
