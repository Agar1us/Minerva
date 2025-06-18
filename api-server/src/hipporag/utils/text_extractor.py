import pathlib
from typing import Callable, Dict, Iterable, List, Union

import os
import csv
import fireducks.pandas as pd
from PyPDF2 import PdfReader

# --------------------------------------------------------------------------- #
# === Per-extension text extractors ========================================= #
# --------------------------------------------------------------------------- #
def _extract_txt(path: pathlib.Path) -> List[str]:
    """
    Parses a document (TXT) to extract pure text.

    :param path: Path to a .txt file.
    :return:     List with a single string – whole file content.
    """
    with path.open("r", encoding="utf-8", errors="ignore") as fp:
        return [fp.read()]

def _extract_csv(path: pathlib.Path) -> List[str]:
    """
    Parses a document (CSV) to extract row-wise texts.

    :param path: Path to .csv file.
    :return:     List of strings – one per *row*.
    """
    rows: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fp:
        reader = csv.reader(fp)
        for i, row in enumerate(reader):
            if i == 0: continue
            text = " ".join(cell for cell in row if cell.strip())
            if text:
                rows.append(text)
    return rows

def _extract_excel(path: pathlib.Path) -> List[str]:
    """
    Parses a document (XLSX, XLS) to extract row-wise texts.

    :param path: Path to an Excel OpenXML file.
    :return:     List of strings – one per *row* across all sheets.
    """
    texts: List[str] = []
    wb = pd.read_excel(path, sheet_name=None, dtype=str, header=None)
    for _, df in wb.items():
        for i, row in enumerate(df.itertuples(index=False)):
            if i == 0: continue
            cells = [str(x) for x in row if pd.notna(x) and str(x).strip()]
            if cells:
                texts.append(" ".join(cells))
    return texts

def _extract_pdf(path: pathlib.Path) -> List[str]:
    """
    Parses a document (PDF) to extract one big text.

    :param path: Path to a PDF file.
    :return:     List with a single string – whole pdf text.
    """
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return ["\n".join(pages)]

# --------------------------------------------------------------------------- #
# === Mapping extension and extractor ========================================#
# --------------------------------------------------------------------------- #
_EXTRACTORS: Dict[str, Callable[[pathlib.Path], str]] = {
    ".txt": _extract_txt,
    ".csv": _extract_csv,
    ".xls": _extract_excel,
    ".xlsx": _extract_excel,
    ".pdf": _extract_pdf,
}

_TABLE_EXTS = {".csv", ".xls", ".xlsx"}  # форматы, где нужен row_index

# --------------------------------------------------------------------------- #
# === Core public API ======================================================= #
# --------------------------------------------------------------------------- #
def collect_texts(paths: Iterable[Union[str, os.PathLike]]) -> List[str]:
    """
    Walks through *,txt/ csv/ xls/ xlsx/ pdf* files contained in *paths* and
    returns a list of strings each consisting of

        absolute_file_path + "#@$" + raw_file_text

    :param paths: An iterable of filesystem paths (files or directories).
    :return:      A flat list of extracted texts (possibly empty).
    """
    results: List[str] = []

    for root_path in map(pathlib.Path, paths):
        if root_path.is_file():
            _handle_file(root_path, results)
        elif root_path.is_dir():
            for dirpath, _, filenames in os.walk(root_path):
                for file_name in filenames:
                    _handle_file(pathlib.Path(dirpath) / file_name, results)
        else:
            # Skip dangling symlinks / non-existent items
            continue

    return results

# --------------------------------------------------------------------------- #
# === Helpers =============================================================== # 
# --------------------------------------------------------------------------- #
def _handle_file(path: pathlib.Path, bucket: List[str]) -> None:
    """
    If *path* has a supported extension, extract its text(s) and append to
    *bucket* in the required format.

    :param path:   The concrete file path.
    :param bucket: Mutable list the results are written into.
    """
    ext = path.suffix.lower()
    extractor = _EXTRACTORS.get(ext)
    if extractor is None:
        return

    try:
        texts = extractor(path)
        abs_path, extension = str(path.resolve()).split('.')

        if ext in _TABLE_EXTS:
            for idx, txt in enumerate(texts, start=1):
                if txt:
                    bucket.append(f"{abs_path}_{idx}.{extension}#@${txt}")
        else:
            for txt in texts:
                if txt:
                    bucket.append(f"{abs_path}#@${txt}")

    except Exception as exc:
        print(f"[WARN] Cannot extract '{path}': {exc}")


if __name__ == "__main__":
    ans = []
    _handle_file(pathlib.Path('test.csv'), ans)
    print(ans)