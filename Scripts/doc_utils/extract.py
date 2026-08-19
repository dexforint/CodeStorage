from collections.abc import Callable
from pathlib import Path

from extract_from_pdf import extract_from_pdf

T = Callable[[Path | str, Path | str | None], None]

ext2fun: dict[str, T] = {
    "pdf": extract_from_pdf,
}


def extract_from_doc(doc_path: Path | str, output_path: Path | str | None):
    extension: str = doc_path.suffix

    assert (
        extension in ext2fun
    ), f"Файл с данным расширением ({extension}) не может быть обработан."

    extract_function: T = ext2fun[extension]

    extract_function(doc_path, output_path)
