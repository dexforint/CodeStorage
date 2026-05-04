import json
from http.server import HTTPServer, BaseHTTPRequestHandler

AUTH_TOKEN = "my_secret_token_123"

# Хранение предыдущего состояния для отслеживания изменений
prev_state = {
    "kills": 0,
    "deaths": 0,
    "assists": 0,
    "health": 100,
    "round_kills": 0,
}


def process_game_state(data: dict):
    """Основная функция обработки игрового состояния"""
    global prev_state

    player = data.get("player", {})
    match_stats = player.get("match_stats", {})
    player_state = player.get("state", {})
    map_info = data.get("map", {})
    round_info = data.get("round", {})
    provider = data.get("provider", {})

    # --- Общая информация ---
    player_name = player.get("name", "Unknown")
    player_team = player.get("team", "N/A")

    # --- Статистика матча ---
    kills = match_stats.get("kills", 0)
    deaths = match_stats.get("deaths", 0)
    assists = match_stats.get("assists", 0)
    mvps = match_stats.get("mvps", 0)
    score = match_stats.get("score", 0)

    # --- Состояние игрока ---
    health = player_state.get("health", 0)
    armor = player_state.get("armor", 0)
    round_kills = player_state.get("round_kills", 0)
    round_killhs = player_state.get("round_killhs", 0)
    money = player_state.get("money", 0)
    flashed = player_state.get("flashed", 0)

    # --- Информация о карте/матче ---
    map_name = map_info.get("name", "N/A")
    map_phase = map_info.get("phase", "N/A")  # warmup, live, intermission, gameover
    ct_score = map_info.get("team_ct", {}).get("score", 0)
    t_score = map_info.get("team_t", {}).get("score", 0)

    # --- Информация о раунде ---
    round_phase = round_info.get("phase", "N/A")  # freezetime, live, over
    bomb_state = round_info.get("bomb", "N/A")  # planted, exploded, defused
    round_winner = round_info.get("win_team", None)

    # ================================================
    # 🎯 ОТСЛЕЖИВАНИЕ СОБЫТИЙ (сравнение с прошлым состоянием)
    # ================================================

    if kills > prev_state["kills"]:
        hs_text = " (HEADSHOT! 🎯)" if round_killhs > 0 else ""
        print(
            f"[KILL] {player_name} получил KILL{hs_text}! "
            f"Всего убийств: {kills} | Убийств в раунде: {round_kills}"
        )

    if deaths > prev_state["deaths"]:
        print(f"[DEATH] {player_name} умер! Всего смертей: {deaths}")

    if assists > prev_state["assists"]:
        print(f"[ASSIST] {player_name} получил ASSIST! Всего ассистов: {assists}")

    if health < prev_state["health"] and health > 0:
        damage = prev_state["health"] - health
        print(
            f"[DAMAGE] {player_name} получил урон -{damage} HP. "
            f"Осталось HP: {health}"
        )

    if health > 0 and prev_state["health"] == 0:
        print(f"[RESPAWN] {player_name} возродился!")

    if round_winner:
        print(
            f"\n[ROUND END] Победила команда: {round_winner} | "
            f"Счёт: CT {ct_score} - {t_score} T\n"
        )

    if bomb_state == "planted":
        print("[BOMB] 💣 Бомба заложена!")
    elif bomb_state == "defused":
        print("[BOMB] ✅ Бомба обезврежена!")
    elif bomb_state == "exploded":
        print("[BOMB] 💥 Бомба взорвалась!")

    # Обновляем прошлое состояние
    prev_state.update(
        {
            "kills": kills,
            "deaths": deaths,
            "assists": assists,
            "health": health,
            "round_kills": round_kills,
        }
    )

    # --- Полный дашборд (выводится каждый апдейт) ---
    print(
        f"[STATE] {player_name} ({player_team}) | "
        f"HP: {health} | Броня: {armor} | 💰 {money}$ | "
        f"K/D/A: {kills}/{deaths}/{assists} | "
        f"Карта: {map_name} [{map_phase}] | "
        f"Раунд: {round_phase}"
    )


class GSIHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Читаем тело запроса
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            return

        # Проверка токена авторизации
        token = data.get("auth", {}).get("token", "")
        if token != AUTH_TOKEN:
            print("[WARNING] Неверный токен авторизации!")
            self.send_response(403)
            self.end_headers()
            return

        print("data received:")
        print("==============")
        print(data)
        print("==============")
        # Обрабатываем данные
        process_game_state(data)

        # Отвечаем игре 200 OK
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        # Отключаем стандартные логи сервера (они засоряют вывод)
        pass


if __name__ == "__main__":
    HOST = "127.0.0.1"
    PORT = 3000

    print(f"🎮 CS2 GSI сервер запущен на http://{HOST}:{PORT}")
    print("Ожидание данных от игры...\n")

    server = HTTPServer((HOST, PORT), GSIHandler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nСервер остановлен.")
        server.server_close()
