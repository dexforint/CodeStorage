from pathlib import Path
from ebooklib import epub, ITEM_DOCUMENT
from html2text import html2text
from clean_epub_markdown import groom_markdown


def extract_from_epub(file_path: Path | str, output_path: Path | str | None = None):
    file_path: Path = Path(file_path)
    extension: str | None = file_path.suffix

    assert extension.lower() == ".epub", f"Расширение ({extension}) должно быть EPUB!"

    file_name: str = file_path.stem
    if output_path is None:
        output_path = file_path.parent / f"{file_name}.md"

    output_path = Path(output_path)

    if output_path.is_dir():
        output_path = output_path / f"{file_name}.md"

    book = epub.read_epub(file_path)

    md_elements: list[str] = []

    for item in book.get_items_of_type(ITEM_DOCUMENT):
        md: str = html2text(item.get_content().decode("utf-8"))
        md_elements.append(md)

    md_text: str = "\n".join(md_elements)

    output_path_raw: Path = output_path.parent / f"{output_path.stem}_raw.md"

    with open(output_path_raw, "w", encoding="utf-8") as file:
        file.write(md_text)

    md_text: str = groom_markdown(md_text)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(md_text)


if __name__ == "__main__":
    input_path = input("Введите путь до EPUB файла: ").replace('"', "")
    extract_from_epub(input_path)
