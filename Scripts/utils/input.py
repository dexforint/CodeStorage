from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory

# Лучше всего хранить файл истории в домашней папке пользователя (как это делает bash или python)
# history_file = os.path.expanduser("~/.my_script_history.txt")

# Подключаем файловую историю
# history = FileHistory(history_file)

# InMemoryHistory сохраняет введенные команды во время работы скрипта,
# чтобы можно было листать их стрелками вверх/вниз.
history = InMemoryHistory()


while True:
    # Используем prompt вместо input
    text = prompt("Введите команду: ", history=history)

    if text.lower() == "exit":
        break

    print(f"Обработка: {text}")
