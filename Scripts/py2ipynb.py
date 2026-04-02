import json
import sys


def py_to_ipynb(py_path, ipynb_path=None):
    if ipynb_path is None:
        ipynb_path = py_path.rsplit(".", 1)[0] + ".ipynb"

    with open(py_path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    # Разбиваем на ячейки по маркерам # %%
    cells_raw = []  # список (тип, [строки])
    current_type = None
    current_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped == "# %%" or stripped.startswith("# %% ["):
            # Сохраняем предыдущую ячейку
            if current_type is not None:
                cells_raw.append((current_type, current_lines))

            current_type = "markdown" if "[markdown]" in stripped else "code"
            current_lines = []
        else:
            if current_type is not None:
                current_lines.append(line)

    # Последняя ячейка
    if current_type is not None:
        cells_raw.append((current_type, current_lines))

    # Собираем notebook-ячейки
    nb_cells = []
    for cell_type, cell_lines in cells_raw:
        # Убираем пустые строки в начале и конце
        while cell_lines and cell_lines[0].strip() == "":
            cell_lines = cell_lines[1:]
        while cell_lines and cell_lines[-1].strip() == "":
            cell_lines = cell_lines[:-1]

        if not cell_lines:
            cell_lines = [""]

        if cell_type == "markdown":
            # Снимаем префикс "# " с каждой строки
            md_lines = []
            for line in cell_lines:
                if line.startswith("# "):
                    md_lines.append(line[2:])
                elif line.strip() == "#":
                    md_lines.append("")
                else:
                    md_lines.append(line)

            source = [l + "\n" for l in md_lines[:-1]] + [md_lines[-1]]

            nb_cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": source,
                }
            )
        else:
            source = [l + "\n" for l in cell_lines[:-1]] + [cell_lines[-1]]

            nb_cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": source,
                }
            )

    notebook = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with open(ipynb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"Converted: {py_path} -> {ipynb_path}")


if __name__ == "__main__":
    # python py2ipynb.py ./data/py2ipynb_example.py ./data/output.ipynb
    if len(sys.argv) < 2:
        print("Usage: python py2ipynb.py input.py [output.ipynb]")
        sys.exit(1)

    py_to_ipynb(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
