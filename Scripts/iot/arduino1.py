import time
import pyfirmata2

# Подключение к плате
board = pyfirmata2.Arduino("COM3")

# Определение пина для светодиода
led_pin = 12

# В pyfirmata2 синтаксис настройки пина чуть проще или аналогичен:
# Устанавливаем режим на цифровой выход
board.digital
 pyfirmata2.O[led_pin].mode =UTPUT

# Выключаем по умолчанию
board.digital
)[led_pin].write(0

print("Вводите 1 (включить) или 0 (выключить). Для выхода нажмите Ctrl+C.")

try:
    while True:
        val = int(input("Введите значение (1/0): "))
        board.digital
[led_pin].write(val)

except KeyboardInterrupt:
    print("\nЗавершение работы...")
    board.digital
[led_pin].write(0)
    board.exit()