import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

AUTH_TOKEN = "my_secret_token_123"

# Хранение предыдущего состояния для отслеживания изменений
prev_state = {
    # Hero
    "health": 100,
    "health_percent": 100,
    "mana": 0,
    "mana_percent": 0,
    "level": 1,
    "alive": True,
    "respawn_seconds": 0,
    "buyback_cooldown": 0,
    "xp": 0,
    # Player
    "kills": 0,
    "deaths": 0,
    "assists": 0,
    "last_hits": 0,
    "denies": 0,
    "gold": 0,
    "gpm": 0,
    "xpm": 0,
    # Map
    "game_state": "DOTA_GAMERULES_STATE_WAIT_FOR_PLAYERS_TO_LOAD",
    "clock_time": 0,
    "roshan_state": "",
}


def format_time(seconds: int) -> str:
    """Форматируем игровое время в MM:SS"""
    if seconds < 0:
        return f"-{abs(seconds) // 60:02}:{abs(seconds) % 60:02}"
    return f"{seconds // 60:02}:{seconds % 60:02}"


def process_hero(hero: dict, player_name: str):
    """Обработка изменений героя"""
    global prev_state

    health = hero.get("health", prev_state["health"])
    health_percent = hero.get("health_percent", prev_state["health_percent"])
    mana = hero.get("mana", prev_state["mana"])
    mana_percent = hero.get("mana_percent", prev_state["mana_percent"])
    level = hero.get("level", prev_state["level"])
    alive = hero.get("alive", prev_state["alive"])
    respawn_seconds = hero.get("respawn_seconds", 0)
    buyback_cooldown = hero.get("buyback_cooldown", 0)
    xp = hero.get("xp", prev_state["xp"])
    hero_name = hero.get("name", "unknown").replace("npc_dota_hero_", "")

    # --- Урон ---
    if health < prev_state["health"] and alive and prev_state["alive"]:
        damage = prev_state["health"] - health
        print(
            f"[HIT]    {player_name} получил -{damage} HP "
            f"({health_percent}% HP осталось)"
        )

    # --- Лечение ---
    if health > prev_state["health"] and alive and prev_state["alive"]:
        heal = health - prev_state["health"]
        print(
            f"[HEAL]   {player_name} восстановил +{heal} HP " f"({health_percent}% HP)"
        )

    # --- Смерть ---
    if not alive and prev_state["alive"]:
        print(
            f"[DEATH]  💀 {player_name} ({hero_name}) умер! "
            f"Возрождение через: {respawn_seconds} сек."
        )

    # --- Возрождение ---
    if alive and not prev_state["alive"]:
        print(f"[SPAWN]  ✅ {player_name} ({hero_name}) возродился!")

    # --- Повышение уровня ---
    if level > prev_state["level"]:
        print(
            f"[LEVEL]  ⬆️  {player_name} ({hero_name}) "
            f"достиг {level} уровня! XP: {xp}"
        )

    # --- Откат бэкпорта ---
    if buyback_cooldown == 0 and prev_state["buyback_cooldown"] > 0:
        print(f"[BUYBACK] 💰 Бэкпорт {player_name} снова доступен!")

    prev_state.update(
        {
            "health": health,
            "health_percent": health_percent,
            "mana": mana,
            "mana_percent": mana_percent,
            "level": level,
            "alive": alive,
            "respawn_seconds": respawn_seconds,
            "buyback_cooldown": buyback_cooldown,
            "xp": xp,
        }
    )

    return hero_name, alive


def process_player(player: dict):
    """Обработка изменений статистики игрока"""
    global prev_state

    player_name = player.get("name", "Unknown")
    kills = player.get("kills", prev_state["kills"])
    deaths = player.get("deaths", prev_state["deaths"])
    assists = player.get("assists", prev_state["assists"])
    last_hits = player.get("last_hits", prev_state["last_hits"])
    denies = player.get("denies", prev_state["denies"])
    gold = player.get("gold", prev_state["gold"])
    gpm = player.get("gpm", prev_state["gpm"])
    xpm = player.get("xpm", prev_state["xpm"])
    team_name = player.get("team_name", "N/A")  # radiant / dire

    # --- Kill ---
    if kills > prev_state["kills"]:
        print(
            f"[KILL]   ⚔️  {player_name} получил KILL! "
            f"Всего: {kills}/{deaths}/{assists}"
        )

    # --- Death (через player) ---
    if deaths > prev_state["deaths"]:
        print(f"[DEATH]  💀 {player_name} погиб! " f"Всего смертей: {deaths}")

    # --- Assist ---
    if assists > prev_state["assists"]:
        print(
            f"[ASSIST] 🤝 {player_name} получил ASSIST! "
            f"Всего: {kills}/{deaths}/{assists}"
        )

    # --- Last Hit ---
    if last_hits > prev_state["last_hits"]:
        lh_diff = last_hits - prev_state["last_hits"]
        print(
            f"[LH]     🗡️  Last Hit x{lh_diff}! "
            f"Всего LH: {last_hits} | DN: {denies}"
        )

    # --- Deny ---
    if denies > prev_state["denies"]:
        print(f"[DN]     🛡️  Deny! Всего денаев: {denies}")

    # --- Изменение золота ---
    if abs(gold - prev_state["gold"]) > 50:  # фильтруем мелкие пассивные изменения
        gold_diff = gold - prev_state["gold"]
        sign = "+" if gold_diff > 0 else ""
        print(f"[GOLD]   💰 {sign}{gold_diff} золота | " f"Итого: {gold} | GPM: {gpm}")

    prev_state.update(
        {
            "kills": kills,
            "deaths": deaths,
            "assists": assists,
            "last_hits": last_hits,
            "denies": denies,
            "gold": gold,
            "gpm": gpm,
            "xpm": xpm,
        }
    )

    return player_name, team_name


def process_map(map_info: dict):
    """Обработка состояния карты"""
    global prev_state

    game_state = map_info.get("game_state", prev_state["game_state"])
    clock_time = map_info.get("clock_time", prev_state["clock_time"])
    map_name = map_info.get("name", "N/A")
    daytime = map_info.get("daytime", True)
    roshan_state = map_info.get("roshan_state", prev_state["roshan_state"])

    # --- Смена состояния игры ---
    if game_state != prev_state["game_state"]:
        state_labels = {
            "DOTA_GAMERULES_STATE_WAIT_FOR_PLAYERS_TO_LOAD": "⏳ Загрузка игроков",
            "DOTA_GAMERULES_STATE_HERO_SELECTION": "🧙 Выбор героев",
            "DOTA_GAMERULES_STATE_STRATEGY_TIME": "📋 Время стратегии",
            "DOTA_GAMERULES_STATE_PRE_GAME": "⚡ Пре-гейм",
            "DOTA_GAMERULES_STATE_GAME_IN_PROGRESS": "🎮 Игра идёт",
            "DOTA_GAMERULES_STATE_POST_GAME": "🏁 Конец игры",
        }
        label = state_labels.get(game_state, game_state)
        print(f"\n[MAP]    🗺️  Состояние матча: {label}")

    # --- Смена дня/ночи ---
    prev_daytime = prev_state.get("daytime", True)
    if daytime != prev_daytime:
        icon = "☀️ " if daytime else "🌙"
        print(
            f"[MAP]    {icon} Наступил {'день' if daytime else 'ночь'}! "
            f"Время: {format_time(clock_time)}"
        )

    # --- Рошан ---
    if roshan_state != prev_state["roshan_state"]:
        roshan_labels = {
            "alive": "🟢 Рошан жив!",
            "respawning_soon": "⚠️  Рошан скоро возродится!",
            "respawning_uncertain": "❓ Рошан может возродиться!",
        }
        label = roshan_labels.get(roshan_state, f"Рошан: {roshan_state}")
        print(f"[ROSHAN] {label}")

    prev_state.update(
        {
            "game_state": game_state,
            "clock_time": clock_time,
            "daytime": daytime,
            "roshan_state": roshan_state,
        }
    )

    return clock_time, map_name, game_state


def process_abilities(abilities: dict, player_name: str):
    """Отслеживание ультимейта"""
    ult = abilities.get("ability5", {})  # 5-й слот — обычно ультимейт

    can_cast = ult.get("can_cast", False)
    cooldown = ult.get("cooldown", 0)
    ult_name = ult.get("name", "").replace("_", " ").title()

    if can_cast and cooldown == 0:
        pass  # не спамим, только при изменении (обрабатывать через prev_state)


def process_game_state(data: dict):
    """Точка входа обработки снимка состояния"""
    # --- previously / added — дельта изменений от Dota ---
    # previously содержит старые значения изменённых полей
    # added      содержит поля, которых не было в прошлом снимке
    previously = data.get("previously", {})
    added = data.get("added", {})

    map_info = data.get("map", {})
    player = data.get("player", {})
    hero = data.get("hero", {})
    abilities = data.get("abilities", {})

    # Обрабатываем только если есть данные игрока
    if not player:
        return

    # --- Карта ---
    clock_time, map_name, game_state = process_map(map_info)

    # --- Игрок ---
    player_name, team_name = process_player(player)

    # --- Герой ---
    if hero:
        hero_name, alive = process_hero(hero, player_name)
    else:
        hero_name = "N/A"
        alive = True

    # --- Способности ---
    if abilities:
        process_abilities(abilities, player_name)

    # --- Итоговый дашборд ---
    hero_data = data.get("hero", {})
    player_data = data.get("player", {})
    health = hero_data.get("health", 0)
    hp_pct = hero_data.get("health_percent", 0)
    mana = hero_data.get("mana", 0)
    mp_pct = hero_data.get("mana_percent", 0)
    level = hero_data.get("level", 1)
    gold = player_data.get("gold", 0)
    kills = player_data.get("kills", 0)
    deaths = player_data.get("deaths", 0)
    assists = player_data.get("assists", 0)
    last_hits = player_data.get("last_hits", 0)
    gpm = player_data.get("gpm", 0)
    xpm = player_data.get("xpm", 0)

    print(
        f"[STATE]  🧙 {player_name} ({hero_name}) [{team_name}] | "
        f"Время: {format_time(clock_time)} | "
        f"HP: {health} ({hp_pct}%) | MP: {mana} ({mp_pct}%) | "
        f"Lvl: {level} | 💰 {gold}g | "
        f"K/D/A: {kills}/{deaths}/{assists} | "
        f"LH: {last_hits} | GPM: {gpm} | XPM: {xpm}"
    )


class GSIHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        # ✅ Отвечаем НЕМЕДЛЕННО — до обработки данных
        # Это важно: игра ждёт 2XX ответа перед отправкой следующего снимка
        self.send_response(200)
        self.end_headers()

        threading.Thread(target=self._process, args=(body,), daemon=True).start()

    def _process(self, body: bytes):
        try:
            data = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return

        # Проверка токена
        token = data.get("auth", {}).get("token", "")
        if token != AUTH_TOKEN:
            print("[WARNING] Неверный токен!")
            return

        process_game_state(data)

    def log_message(self, format, *args):
        pass  # Отключаем стандартные логи


if __name__ == "__main__":
    HOST = "127.0.0.1"
    PORT = 3000

    print(f"🎮 Dota 2 GSI сервер запущен на http://{HOST}:{PORT}")
    print("Ожидание данных от игры...\n")

    server = HTTPServer((HOST, PORT), GSIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nСервер остановлен.")
        server.server_close()
