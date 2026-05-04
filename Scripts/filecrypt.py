#!/usr/bin/env python3
"""
Шифрование и дешифрование файлов паролем.

Использование:
    python filecrypt.py secret.txt          # → создаст secret.txt.encrypted
    python filecrypt.py secret.txt.encrypted # → создаст secret.txt
"""

import argparse
import os
import sys
import base64
from getpass import getpass
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

EXTENSION = ".encrypted"
SALT_SIZE = 16
KDF_ITERATIONS = 600_000


def derive_key(password: str, salt: bytes) -> bytes:
    """Генерация криптографического ключа из пароля."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=KDF_ITERATIONS,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))


def encrypt_file(filepath: Path, password: str) -> Path:
    """Шифрует файл, возвращает путь к зашифрованному файлу."""
    output_path = filepath.with_name(filepath.name + EXTENSION)

    if output_path.exists():
        raise FileExistsError(f"Файл уже существует: {output_path}")

    salt = os.urandom(SALT_SIZE)
    key = derive_key(password, salt)
    fernet = Fernet(key)

    original_data = filepath.read_bytes()
    encrypted_data = fernet.encrypt(original_data)

    output_path.write_bytes(salt + encrypted_data)

    return output_path


def decrypt_file(filepath: Path, password: str) -> Path:
    """Расшифровывает файл, возвращает путь к расшифрованному файлу."""
    # Убираем .encrypted из имени
    if not filepath.name.endswith(EXTENSION):
        raise ValueError(f"Файл не имеет расширения {EXTENSION}")

    output_name = filepath.name[: -len(EXTENSION)]
    output_path = filepath.with_name(output_name)

    if output_path.exists():
        raise FileExistsError(f"Файл уже существует: {output_path}")

    file_data = filepath.read_bytes()

    if len(file_data) < SALT_SIZE:
        raise ValueError("Файл повреждён или не является зашифрованным")

    salt = file_data[:SALT_SIZE]
    encrypted_data = file_data[SALT_SIZE:]

    key = derive_key(password, salt)
    fernet = Fernet(key)

    try:
        decrypted_data = fernet.decrypt(encrypted_data)
    except InvalidToken:
        raise ValueError("Неверный пароль или файл повреждён")

    output_path.write_bytes(decrypted_data)

    return output_path


def ask_password(confirm: bool = False) -> str:
    """Запрашивает пароль у пользователя."""
    password = getpass("🔑 Введите пароль: ")

    if not password:
        print("❌ Пароль не может быть пустым", file=sys.stderr)
        sys.exit(1)

    if confirm:
        password2 = getpass("🔑 Подтвердите пароль: ")
        if password != password2:
            print("❌ Пароли не совпадают", file=sys.stderr)
            sys.exit(1)

    return password


def main():
    parser = argparse.ArgumentParser(
        description="Шифрование/дешифрование файла паролем"
    )
    parser.add_argument(
        "file",
        type=Path,
        help=(
            f"путь к файлу: если расширение {EXTENSION} — расшифровать, "
            f"иначе — зашифровать"
        ),
    )
    args = parser.parse_args()

    filepath: Path = args.file

    # --- Проверяем, что файл существует ---
    if not filepath.is_file():
        print(f"❌ Файл не найден: {filepath}", file=sys.stderr)
        sys.exit(1)

    # --- Определяем действие по расширению ---
    need_decrypt = filepath.name.endswith(EXTENSION)

    if need_decrypt:
        print(f"🔓 Расшифровка: {filepath}")
        password = ask_password(confirm=False)

        try:
            result = decrypt_file(filepath, password)
        except (ValueError, FileExistsError) as e:
            print(f"❌ {e}", file=sys.stderr)
            sys.exit(1)

        print(f"✅ Расшифрованный файл: {result}")

    else:
        print(f"🔒 Шифрование: {filepath}")
        password = ask_password(confirm=True)  # при шифровании — подтверждение

        try:
            result = encrypt_file(filepath, password)
        except FileExistsError as e:
            print(f"❌ {e}", file=sys.stderr)
            sys.exit(1)

        print(f"✅ Зашифрованный файл: {result}")


if __name__ == "__main__":
    main()
