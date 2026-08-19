import pymupdf4llm
from pathlib import Path
from clean_pdf_markdown import PdfMarkdownCleaner


def extract_from_pdf(pdf_file_path: Path | str, output_path: Path | str | None = None):
    pdf_file_path: Path = Path(pdf_file_path)
    pdf_name: str = pdf_file_path.stem

    if output_path is None:
        output_path = pdf_file_path.parent / f"{pdf_name}.md"

    output_path = Path(output_path)

    if output_path.is_dir():
        output_path = output_path / f"{pdf_name}.md"

    md_text: str = pymupdf4llm.to_markdown(pdf_file_path, use_ocr=False)

    cleaner = PdfMarkdownCleaner()
    md_text = cleaner.clean(md_text)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(md_text)


if __name__ == "__main__":
    extract_from_pdf(input("Введите путь до PDF файла: "), "./data")
