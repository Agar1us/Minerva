import os
import pathlib
from typing import Dict, Iterable, List, Union

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    ApiVlmOptions,
    ResponseFormat,
    VlmPipelineOptions,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    ImageFormatOption,
)
from docling.pipeline.vlm_pipeline import VlmPipeline


# --------------------------------------------------------------------------- #
# === Converter configuration =============================================== #
# --------------------------------------------------------------------------- #

def build_vlm_options(model: str, prompt: str) -> VlmPipelineOptions:
    api_opts = ApiVlmOptions(
        url=os.getenv('OCR_MODEL_URL'),
        timeout=90,
        scale=1.0,
        response_format=ResponseFormat.MARKDOWN,
        params=dict(model=model),
        prompt=prompt,
    )
    vlm_opts = VlmPipelineOptions(enable_remote_services=True)
    vlm_opts.vlm_options = api_opts
    return vlm_opts


PROMPT = (
    "Extract the text from the above document as if you were reading it naturally. "
    "Return the tables in html format. Return the equations in LaTeX representation. "
    "If there is an image in the document and image caption is not present, add a small "
    "description of the image inside the <img></img> tag; otherwise, add the image caption "
    "inside <img></img>. Watermarks should be wrapped in brackets. "
    "Page numbers should be wrapped in brackets. Prefer using ☐ and ☑ for check boxes."
)

vlm_opts = build_vlm_options(model=os.getenv('OCR_MODEL_NAME'), prompt=PROMPT)

FORMAT_OPTIONS: Dict[InputFormat, object] = {
    InputFormat.PDF: PdfFormatOption(
        pipeline_cls=VlmPipeline,
        pipeline_options=vlm_opts,
    ),
    InputFormat.IMAGE: ImageFormatOption(
        pipeline_cls=VlmPipeline,
        pipeline_options=vlm_opts,
    ),
}

CONVERTER = DocumentConverter(format_options=FORMAT_OPTIONS)

_TABLE_FORMATS = {InputFormat.CSV, InputFormat.XLSX}


# --------------------------------------------------------------------------- #
# === Core public API ======================================================= #
# --------------------------------------------------------------------------- #
def collect_texts(paths: Iterable[Union[str, os.PathLike]]) -> List[str]:
    """
    Recursively walks through the supplied *paths* and returns a flat list
    of strings in the format

        <abs_path>[ _<row_idx>].<ext>#@$<raw_text>

    :param paths: Iterable with file or directory paths.
    :return:      List with extracted texts (can be empty).
    """
    bucket: List[str] = []

    for root in map(pathlib.Path, paths):
        if root.is_file():
            _handle_file(root, bucket)
        elif root.is_dir():
            for dirpath, _, filenames in os.walk(root):
                for fname in filenames:
                    _handle_file(pathlib.Path(dirpath) / fname, bucket)

    return bucket


# --------------------------------------------------------------------------- #
# === Helpers =============================================================== # 
# --------------------------------------------------------------------------- #

def extract_table_rows(table_item):
    """
    Extract rows from a TableItem object.
    
    :param table_item: A TableItem object containing table data
    :return: A list of lists, where each inner list represents a row of text cells
    """
    cells = table_item.data.table_cells
    row_indices = sorted(set(cell.start_row_offset_idx for cell in cells))
    rows = []
    for row_idx in row_indices:
        row_cells = [cell for cell in cells if cell.start_row_offset_idx == row_idx]
        row_cells.sort(key=lambda cell: cell.start_col_offset_idx)
        row_texts = [cell.text for cell in row_cells if cell.text]
        rows.append(row_texts)
    return rows

def _handle_file(path: pathlib.Path, bucket: List[str]) -> None:
    """
    Converts *path* with docling and appends the result(s) to *bucket*
    if the extension is supported.

    :param path:   Concrete file path.
    :param bucket: Mutable list used as an output accumulator.
    """
    try:
        result = CONVERTER.convert(path)
    except Exception as exc:
        print(f"[WARN] Cannot convert '{path}': {exc}")
        return

    doc = result.document
    ext = path.suffix.lstrip(".")

    if path.suffix.lower()[1:] in _TABLE_FORMATS:
        for table in doc.tables:
            table_rows = extract_table_rows(table)
            for idx, row in enumerate(table_rows[1:], start=1):
                row_text = " ".join(str(cell).strip() for cell in row if cell and str(cell).strip())
                bucket.append(f"{path.stem}_{idx}.{ext}#@${row_text}")
    else:
        text = doc.export_to_text().strip()
        if text:
            bucket.append(f"{str(path)}#@${text}")
