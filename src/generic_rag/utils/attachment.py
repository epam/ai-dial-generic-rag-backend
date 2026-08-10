import io

from aidial_sdk.chat_completion import Attachment
from datauri import DataURI
from PIL import Image
from PIL.Image import Resampling
from pydantic import StrictStr

from generic_rag.types import AnyChunk, ChunkSource, ImageChunk, TextChunk


def create_attachment(chunks: list[AnyChunk], cite_index: str | int) -> Attachment | None:
    if not chunks:
        return None

    data = ""
    title = ""
    reference_url = ""

    for chunk in chunks:
        if not (title and reference_url):
            chunk_source = ChunkSource.from_chunk(chunk)
            title = f"[{cite_index}] {chunk_source.source_display_name}"
            reference_url = (
                f"{chunk_source.source_url}#page={chunk.page_number}"
                if chunk.page_number
                else chunk_source.source_url
            )

        if isinstance(chunk, TextChunk):
            data += f"{chunk.text}\n\n"
        if isinstance(chunk, ImageChunk):
            image_title = f"Image of page #{chunk.page_number}"
            image_url = create_thumbnail(chunk)
            data += f'![{image_title}]({image_url} "{image_title}")\n\n'

    return Attachment(
        type=StrictStr("text/markdown"),
        title=StrictStr(title),
        data=StrictStr(data.rstrip() or " "),
        reference_url=StrictStr(reference_url),
    )


def create_thumbnail(chunk: ImageChunk, size: int = 256) -> str:
    """
    Create thumbnail for given image chunk.

    :param chunk: the chunk with original image
    :param size: requested thumbnail size in pixels
    :return: base64-encoded data url for created thumbnail
    """
    chunk_image = Image.open(io.BytesIO(chunk.content))
    chunk_image.thumbnail(size=(size, size), resample=Resampling.BICUBIC)
    with io.BytesIO() as fp:
        chunk_image.save(fp, format="jpeg")
        return DataURI.make(
            mimetype="image/jpeg",
            charset=None,
            base64=True,
            data=fp.getvalue(),
        )
