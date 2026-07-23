# https://arena.ai/c/019f8e17-e364-7df5-893e-32e2e7c2eeb2
import subprocess
import os
import sys

# === SETTINGS ===
MAX_DIFF_LINES = 300  # Maximum number of lines per file in the diff

# Files to include in the summary but exclude from the detailed diff
EXCLUDE_FROM_DIFF = [
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
    "Cargo.lock",
    "*.min.js",
    "*.min.css",
]


def run_command(command, cwd, env=None):
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            env=env,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        if command[1] == "diff" and e.returncode == 1:
            return e.stdout.strip()
        return ""
    except Exception:
        return ""


def find_git_root(path):
    """Returns the absolute path to the root of the git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            check=True,
        )
        return os.path.normpath(result.stdout.strip())
    except subprocess.CalledProcessError:
        return None


def copy_to_clipboard(text):
    """Copies text to Windows clipboard using the clip utility."""
    try:
        subprocess.run(["clip"], input=text, text=True, encoding="utf-8", check=True)
        return True
    except Exception as e:
        print(f"Error copying to clipboard: {e}")
        return False


def clean_and_truncate_diff(diff_text, max_lines):
    """Cleans Git metadata noise and truncates large diff chunks."""
    if not diff_text:
        return ""

    lines = diff_text.split("\n")
    out_lines = []
    current_file_lines = 0
    is_truncating = False

    for line in lines:
        # --- CLEANING GIT NOISE ---
        # Skip index hashes
        if line.startswith("index "):
            continue
        # Skip file mode changes
        if line.startswith("new file mode ") or line.startswith("deleted file mode "):
            continue
        if line.startswith("old mode ") or line.startswith("new mode "):
            continue
        # Skip redundant path markers (diff --git already has the path)
        if line.startswith("--- ") or line.startswith("+++ "):
            continue

        # --- TRUNCATION LOGIC ---
        if line.startswith("diff --git "):
            current_file_lines = 0
            is_truncating = False
            out_lines.append(line)
            continue

        current_file_lines += 1

        if is_truncating:
            continue

        if current_file_lines > max_lines:
            out_lines.append(
                f"\n... [WARNING: Code truncated. Exceeded limit of {max_lines} lines for this file] ...\n"
            )
            is_truncating = True
        else:
            out_lines.append(line)

    return "\n".join(out_lines)


def get_uncommitted_changes(target_path, output_file, copy_to_clip=False):
    if not os.path.isdir(target_path):
        print(f"\nError: Directory {target_path} does not exist.")
        sys.exit(1)

    # 1. Find git repository root
    repo_root = find_git_root(target_path)
    if not repo_root:
        print(f"\nError: Directory {target_path} is not inside a Git repository.")
        sys.exit(1)

    # 2. Calculate relative path (for git diff filtering)
    rel_target_path = os.path.relpath(target_path, repo_root)
    rel_target_path = rel_target_path.replace("\\", "/")

    print(f"\nRepository root: {repo_root}")
    if rel_target_path != ".":
        print(f"Filtering changes by directory: {rel_target_path}")

    git_dir = os.path.join(repo_root, ".git")
    env = os.environ.copy()
    temp_index = os.path.join(git_dir, "temp_index_llm")
    env["GIT_INDEX_FILE"] = temp_index

    output_content = ["Here are the current uncommitted changes in the project.\n"]
    if rel_target_path != ".":
        output_content[0] = (
            f"Here are the current uncommitted changes in the directory '{rel_target_path}'.\n"
        )

    try:
        has_commits = run_command(["git", "rev-parse", "HEAD"], cwd=repo_root)
        if has_commits:
            run_command(["git", "read-tree", "HEAD"], cwd=repo_root, env=env)
        run_command(["git", "add", "-A"], cwd=repo_root, env=env)

        # 3. SUMMARY OF CHANGES
        summary_cmd = [
            "git",
            "-c",
            "core.quotepath=false",
            "diff",
            "--cached",
            "HEAD",
            "-M",
            "--name-status",
            "--",
            rel_target_path,
        ]
        summary = run_command(summary_cmd, cwd=repo_root, env=env)

        if not summary:
            print("No changes found. Working tree is clean.")
            return

        output_content.append("=== SUMMARY OF CHANGES ===")
        for line in summary.split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            status, file1 = parts[0], parts[1]
            file2 = parts[2] if len(parts) > 2 else ""

            if status.startswith("R"):
                output_content.append(f"RENAMED/MOVED: {file1} -> {file2}")
            elif status.startswith("A"):
                output_content.append(f"NEW FILE: {file1}")
            elif status.startswith("D"):
                output_content.append(f"DELETED: {file1}")
            elif status.startswith("M"):
                output_content.append(f"MODIFIED: {file1}")
        output_content.append("\n")

        # 4. DETAILED DIFF
        diff_cmd = [
            "git",
            "-c",
            "core.quotepath=false",
            "diff",
            "--cached",
            "HEAD",
            "-M",
            "--",
            rel_target_path,
        ]

        for ex in EXCLUDE_FROM_DIFF:
            diff_cmd.append(f":(exclude){ex}")

        diff_output = run_command(diff_cmd, cwd=repo_root, env=env)

        diff_output = clean_and_truncate_diff(diff_output, MAX_DIFF_LINES)

        output_content.append("=== DETAILED DIFF ===")
        output_content.append("```diff\n" + diff_output + "\n```\n")

        final_text = "\n".join(output_content)

        # Save to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_text)
        print(f"Done! Changes saved to file: {output_file}")

        # Copy to clipboard
        if copy_to_clip:
            if copy_to_clipboard(final_text):
                print("Success: Text automatically copied to clipboard!")

    finally:
        if os.path.exists(temp_index):
            os.remove(temp_index)


if __name__ == "__main__":
    print("--- Export uncommitted Git changes for LLM ---")

    # 1. Ask for target directory
    repo_input = input(
        r"Enter path to project or directory (Enter for current '.'): "
    ).strip()
    target_path = repo_input if repo_input else "."

    # 2. Ask for output file name
    out_input = input("Output file name (Enter for 'llm_prompt_changes.txt'): ").strip()
    out_path = out_input if out_input else "llm_prompt_changes.txt"

    # 3. Ask for clipboard copy
    # copy_input = input("Copy result to clipboard? [Y/n]: ").strip().lower()
    copy_to_clip = True  # copy_input in ["", "y", "yes"]

    target_abs = os.path.abspath(target_path)
    out_abs = os.path.abspath(out_path)

    get_uncommitted_changes(target_abs, out_abs, copy_to_clip)
