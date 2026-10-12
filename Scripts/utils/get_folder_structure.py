from pathlib import Path


def print_tree(directory, prefix=""):
    try:
        contents = sorted(
            directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
        )
    except PermissionError:
        print(prefix + "[Permission Denied]")
        return

    pointers = ["├── "] * (len(contents) - 1) + ["└── "]
    for pointer, path in zip(pointers, contents):
        print(prefix + pointer + path.name)
        if path.is_dir():
            extension = "│   " if pointer == "├── " else "    "
            print_tree(path, prefix + extension)


path_input = input().strip()
if (
    len(path_input) >= 2
    and path_input[0] == path_input[-1]
    and path_input[0] in ('"', "'")
):
    path_input = path_input[1:-1]

root = Path(path_input)

if not root.exists():
    print("Path does not exist.")
elif not root.is_dir():
    print("Path is not a directory.")
else:
    print(root.name)
    print_tree(root)
