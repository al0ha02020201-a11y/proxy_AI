import asyncio
import datetime
import io
import logging
import os
import re
import sqlite3
import tempfile
import time
from pathlib import Path

import psutil
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)
from google import genai
from google.genai import types as genai_types

# =====================================================================
# ⚙️ НАСТРОЙКИ (ЗАПОЛНИТЕ СВОИ ДАННЫЕ)
# =====================================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "ВАШ_ТОКЕН_ТЕЛЕГРАМ_БОТА")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "ВАШ_КЛЮЧ_GEMINI_API")

# Укажите ваш Telegram ID (узнать можно у @userinfobot), чтобы работал /admin и алерты
ADMIN_IDS = [123456789]

DEFAULT_MODEL = "gemini-2.5-flash"
DB_PATH = "gemini_bot.db"
HEALTH_CHECK_INTERVAL = 600  # 10 минут (в секундах)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Инициализация бота и клиента Gemini
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()
client = genai.Client(api_key=GEMINI_API_KEY)

# Глобальное состояние для мониторинга и кэша
START_TIME = time.time()
MODELS_CACHE = []
HEALTH_STATE = {
    "is_healthy": True,
    "last_error": None,
    "last_ping_ms": 0,
    "last_check": "Еще не проводилась",
    "alerts_enabled": True,
}

# Расширения файлов, которые лучше читать напрямую как текст/код
TEXT_CODE_EXTENSIONS = {
    ".txt", ".py", ".js", ".ts", ".html", ".css", ".json", ".xml", ".yaml", ".yml",
    ".md", ".csv", ".sql", ".sh", ".bat", ".ps1", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".java", ".go", ".rs", ".php", ".rb", ".kt", ".swift", ".ini", ".cfg",
    ".env", ".dockerfile", ".toml", ".log"
}

# Соответствие языков программирования расширениям файлов для выгрузки кода
LANG_EXT_MAP = {
    "python": "py", "py": "py", "javascript": "js", "js": "js",
    "typescript": "ts", "ts": "ts", "html": "html", "css": "css",
    "json": "json", "sql": "sql", "bash": "sh", "sh": "sh",
    "c": "c", "cpp": "cpp", "c++": "cpp", "csharp": "cs", "cs": "cs",
    "java": "java", "go": "go", "rust": "rs", "php": "php",
    "ruby": "rb", "kotlin": "kt", "swift": "swift", "xml": "xml",
    "yaml": "yaml", "markdown": "md", "md": "md"
}


# =====================================================================
# 🗄 БАЗА ДАННЫХ (SQLite)
# =====================================================================
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                current_model TEXT DEFAULT ?,
                active_chat_id INTEGER DEFAULT 0
            )
        """, (DEFAULT_MODEL,))
        c.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                chat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                msg_id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER,
                role TEXT,
                text TEXT,
                file_uri TEXT,
                file_mime TEXT,
                created_at REAL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS stats (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                total_requests INTEGER DEFAULT 0,
                success_requests INTEGER DEFAULT 0,
                error_requests INTEGER DEFAULT 0,
                files_processed INTEGER DEFAULT 0
            )
        """)
        c.execute("INSERT OR IGNORE INTO stats (id) VALUES (1)")
        conn.commit()


def ensure_user(user_id: int) -> tuple[str, int]:
    """Возвращает (current_model, active_chat_id) пользователя. Создает чат, если его нет."""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT current_model, active_chat_id FROM users WHERE user_id = ?", (user_id,))
        row = c.fetchone()
        if not row:
            c.execute("INSERT INTO chats (user_id, title) VALUES (?, ?)", (user_id, "Новый диалог"))
            chat_id = c.lastrowid
            c.execute(
                "INSERT INTO users (user_id, current_model, active_chat_id) VALUES (?, ?, ?)",
                (user_id, DEFAULT_MODEL, chat_id)
            )
            conn.commit()
            return DEFAULT_MODEL, chat_id

        model, chat_id = row
        # Проверяем, существует ли активный чат
        c.execute("SELECT chat_id FROM chats WHERE chat_id = ? AND user_id = ?", (chat_id, user_id))
        if not c.fetchone():
            c.execute("INSERT INTO chats (user_id, title) VALUES (?, ?)", (user_id, "Новый диалог"))
            chat_id = c.lastrowid
            c.execute("UPDATE users SET active_chat_id = ? WHERE user_id = ?", (chat_id, user_id))
            conn.commit()
        return model, chat_id


def create_new_chat(user_id: int) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("INSERT INTO chats (user_id, title) VALUES (?, ?)", (user_id, "Новый диалог"))
        chat_id = c.lastrowid
        c.execute("UPDATE users SET active_chat_id = ? WHERE user_id = ?", (chat_id, user_id))
        conn.commit()
        return chat_id


def get_user_chats(user_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "SELECT chat_id, title, created_at FROM chats WHERE user_id = ? ORDER BY chat_id ASC",
            (user_id,)
        )
        return c.fetchall()


def set_active_chat(user_id: int, chat_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET active_chat_id = ? WHERE user_id = ?", (chat_id, user_id))
        conn.commit()


def set_user_model(user_id: int, model_name: str):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET current_model = ? WHERE user_id = ?", (model_name, user_id))
        conn.commit()


def add_message(chat_id: int, role: str, text: str, file_uri: str = None, file_mime: str = None):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO messages (chat_id, role, text, file_uri, file_mime, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (chat_id, role, text, file_uri, file_mime, time.time())
        )
        # Если это первое сообщение пользователя, обновим название чата
        if role == "user" and text:
            c.execute("SELECT COUNT(*) FROM messages WHERE chat_id = ?", (chat_id,))
            if c.fetchone()[0] <= 1:
                clean_title = text.strip().replace("\n", " ")[:35]
                if len(text) > 35:
                    clean_title += "..."
                c.execute("UPDATE chats SET title = ? WHERE chat_id = ?", (clean_title, chat_id))
        conn.commit()
        return c.lastrowid


def get_chat_messages(chat_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "SELECT role, text, file_uri, file_mime, created_at FROM messages WHERE chat_id = ? ORDER BY msg_id ASC",
            (chat_id,)
        )
        return c.fetchall()


def get_message_by_id(msg_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT text FROM messages WHERE msg_id = ?", (msg_id,))
        row = c.fetchone()
        return row[0] if row else None


def update_stats(req=0, success=0, err=0, files=0):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            UPDATE stats SET
                total_requests = total_requests + ?,
                success_requests = success_requests + ?,
                error_requests = error_requests + ?,
                files_processed = files_processed + ?
            WHERE id = 1
        """, (req, success, err, files))
        conn.commit()


def get_stats():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT total_requests, success_requests, error_requests, files_processed FROM stats WHERE id = 1")
        st = c.fetchone()
        c.execute("SELECT COUNT(*) FROM users")
        users_cnt = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM chats")
        chats_cnt = c.fetchone()[0]
        return {
            "total": st[0], "success": st[1], "errors": st[2], "files": st[3],
            "users": users_cnt, "chats": chats_cnt
        }


# =====================================================================
# 🎹 КЛАВИАТУРЫ И НАВИГАЦИЯ
# =====================================================================
def get_main_keyboard(user_id: int) -> ReplyKeyboardMarkup:
    kb = [
        [
            KeyboardButton(text="⬅️ Прошлый чат"),
            KeyboardButton(text="➕ Новый чат"),
            KeyboardButton(text="➡️ Следующий чат"),
        ],
        [
            KeyboardButton(text="🗂 Все чаты"),
            KeyboardButton(text="🤖 Выбор модели"),
        ]
    ]
    if user_id in ADMIN_IDS:
        kb[1].append(KeyboardButton(text="📊 Админ"))
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)


def get_response_export_kb(db_msg_id: int, has_code: bool) -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(text="📄 Скачать .txt", callback_data=f"export:txt:{db_msg_id}"),
        InlineKeyboardButton(text="📝 Скачать .md", callback_data=f"export:md:{db_msg_id}"),
    ]
    rows = [buttons]
    if has_code:
        rows.append([InlineKeyboardButton(text="💻 Выгрузить код файлом", callback_data=f"export:code:{db_msg_id}")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


# =====================================================================
# 🤖 ПОЛУЧЕНИЕ И СОРТИРОВКА МОДЕЛЕЙ (ОТ НОВЫХ К СТАРЫМ)
# =====================================================================
def parse_model_sort_key(model_name: str):
    """Ключ сортировки: ставит самые свежие версии Gemini и Imagen наверх."""
    # Ищем версию вида 3.0, 2.5, 2.0, 1.5, 1.0
    ver_match = re.search(r"(\d+)\.(\d+)", model_name)
    if ver_match:
        major, minor = int(ver_match.group(1)), int(ver_match.group(2))
    else:
        single_ver = re.search(r"-(\d+)-", model_name)
        major = int(single_ver.group(1)) if single_ver else 0
        minor = 0

    # Приоритет подтипов внутри одной версии
    tier = 1
    if "pro" in model_name:
        tier = 4
    elif "flash" in model_name and "lite" not in model_name:
        tier = 3
    elif "imagen" in model_name:
        tier = 2

    is_latest = 1 if ("latest" in model_name or "preview" in model_name or "exp" in model_name) else 0
    return (major, minor, tier, is_latest, model_name)


async def fetch_sorted_models(force=False) -> list[str]:
    global MODELS_CACHE
    if MODELS_CACHE and not force:
        return MODELS_CACHE

    try:
        raw_models = []
        pager = await client.aio.models.list()
        async for m in pager:
            name = m.name.replace("models/", "")
            # Оставляем только генеративные модели Gemini и Imagen (исключаем технических/embedding)
            if any(x in name for x in ["embedding", "aqa", "bison", "gecko", "learnlm"]):
                continue
            if name.startswith("gemini") or name.startswith("imagen"):
                raw_models.append(name)

        # Сортируем по убыванию (от самых новых к самым старым)
        raw_models = sorted(list(set(raw_models)), key=parse_model_sort_key, reverse=True)
        if raw_models:
            MODELS_CACHE = raw_models
            return MODELS_CACHE
    except Exception as e:
        logger.error(f"Ошибка получения списка моделей: {e}")

    # Резервный список, если API не вернул каталог
    return [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "imagen-3.0-generate-002",
    ]


async def build_models_keyboard(user_id: int, page: int = 0) -> InlineKeyboardMarkup:
    models = await fetch_sorted_models()
    current_model, _ = ensure_user(user_id)

    per_page = 7
    total_pages = max(1, (len(models) + per_page - 1) // per_page)
    page = max(0, min(page, total_pages - 1))

    start_idx = page * per_page
    page_models = models[start_idx : start_idx + per_page]

    rows = []
    for idx, m in enumerate(page_models):
        prefix = "✅ " if m == current_model else ""
        badge = "🔥 " if (start_idx + idx) < 3 else ""
        rows.append([InlineKeyboardButton(text=f"{prefix}{badge}{m}", callback_data=f"setmodel:{m}:{page}")])

    nav_row = []
    if page > 0:
        nav_row.append(InlineKeyboardButton(text="⬅️ Назад", callback_data=f"modelpage:{page - 1}"))
    nav_row.append(InlineKeyboardButton(text=f"Стр. {page + 1}/{total_pages}", callback_data="noop"))
    if page < total_pages - 1:
        nav_row.append(InlineKeyboardButton(text="Вперед ➡️", callback_data=f"modelpage:{page + 1}"))

    rows.append(nav_row)
    rows.append([InlineKeyboardButton(text="🔄 Обновить список из API", callback_data="modelpage:refresh")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


# =====================================================================
# 🚨 ФОНОВЫЙ МОНИТОРИНГ СЕРВЕРА И API (КАЖДЫЕ 10 МИНУТ)
# =====================================================================
async def check_service_health() -> tuple[bool, str, int]:
    """Проверяет доступность Gemini API и ресурсы сервера."""
    t0 = time.monotonic()
    try:
        # Легкий запрос к API для проверки связи
        await asyncio.wait_for(
            client.aio.models.get(model=DEFAULT_MODEL),
            timeout=15.0
        )
        ping_ms = int((time.monotonic() - t0) * 1000)

        cpu = psutil.cpu_percent(interval=0.5)
        ram = psutil.virtual_memory().percent

        if cpu > 95 or ram > 95:
            return False, f"Перегрузка железа! CPU: {cpu}%, RAM: {ram}%", ping_ms

        return True, "OK", ping_ms
    except Exception as e:
        ping_ms = int((time.monotonic() - t0) * 1000)
        return False, f"{type(e).__name__}: {str(e)[:150]}", ping_ms


async def background_health_monitor():
    """Фоновый цикл: каждые 10 минут проверяет сервис и бьет тревогу, если серверу бо-бо."""
    while True:
        await asyncio.sleep(HEALTH_CHECK_INTERVAL)
        ok, details, ping_ms = await check_service_health()
        now_str = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")

        HEALTH_STATE["is_healthy"] = ok
        HEALTH_STATE["last_ping_ms"] = ping_ms
        HEALTH_STATE["last_check"] = now_str
        if not ok:
            HEALTH_STATE["last_error"] = f"[{now_str}] {details}"

        if not ok and HEALTH_STATE["alerts_enabled"]:
            alert_text = (
                "🚨 <b>ВНИМАНИЕ! СЕРВЕРУ БО-БО!</b> 🚨\n\n"
                f"🕒 <b>Время:</b> {now_str}\n"
                f"⚠️ <b>Причина:</b> <code>{details}</code>\n"
                f"📡 <b>Пинг:</b> {ping_ms} мс\n"
                f"💻 <b>CPU:</b> {psutil.cpu_percent()}% | <b>RAM:</b> {psutil.virtual_memory().percent}%"
            )
            for admin_id in ADMIN_IDS:
                try:
                    await bot.send_message(admin_id, alert_text, parse_mode="HTML")
                except Exception as ex:
                    logger.error(f"Не удалось отправить алерт админу {admin_id}: {ex}")


# =====================================================================
# 🛠 АДМИН-ПАНЕЛЬ (/admin)
# =====================================================================
def format_uptime(seconds: float) -> str:
    td = datetime.timedelta(seconds=int(seconds))
    days = td.days
    hours, rem = divmod(td.seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{days}д {hours}ч {minutes}м {secs}с"


async def render_admin_dashboard() -> tuple[str, InlineKeyboardMarkup]:
    ok, details, ping_ms = await check_service_health()
    now_str = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    HEALTH_STATE["is_healthy"] = ok
    HEALTH_STATE["last_ping_ms"] = ping_ms
    HEALTH_STATE["last_check"] = now_str
    if not ok:
        HEALTH_STATE["last_error"] = f"[{now_str}] {details}"

    st = get_stats()
    uptime_str = format_uptime(time.time() - START_TIME)
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    status_Icon = "🟢 РАБОТАЕТ ШТАТНО" if ok else "🔴 СЕРВЕРУ БО-БО (СБОЙ)"
    alerts_status = "🔔 Включены (каждые 10 мин)" if HEALTH_STATE["alerts_enabled"] else "🔕 Выключены"

    text = (
        f"📊 <b>АДМИН-ПАНЕЛЬ И МОНИТОРИНГ</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Статус сервиса:</b> {status_Icon}\n"
        f"<b>Пинг до Gemini API:</b> <code>{ping_ms} мс</code>\n"
        f"<b>Рантайм (Uptime):</b> <code>{uptime_str}</code>\n"
        f"<b>Алерты (10 мин):</b> {alerts_status}\n"
        f"<b>Последняя ошибка:</b> <code>{HEALTH_STATE['last_error'] or 'Нет ошибок'}</code>\n\n"
        f"📈 <b>Статистика запросов:</b>\n"
        f"• Всего запросов: <b>{st['total']}</b>\n"
        f"• Успешных: <b>{st['success']}</b> | Ошибок: <b>{st['errors']}</b>\n"
        f"• Обработано файлов: <b>{st['files']}</b>\n"
        f"• Пользователей: <b>{st['users']}</b> | Чатов: <b>{st['chats']}</b>\n\n"
        f"🖥 <b>Ресурсы сервера:</b>\n"
        f"• CPU: <code>{cpu}%</code>\n"
        f"• RAM: <code>{mem.percent}% ({mem.used // (1024**2)} / {mem.total // (1024**2)} МБ)</code>\n"
        f"• Диск: <code>{disk.percent}%</code>"
    )

    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔄 Обновить показатели", callback_data="admin:refresh")],
        [
            InlineKeyboardButton(
                text="🔕 Выкл алерты" if HEALTH_STATE["alerts_enabled"] else "🔔 Вкл алерты",
                callback_data="admin:toggle_alerts"
            ),
            InlineKeyboardButton(text="🚨 Тест «Серверу бо-бо»", callback_data="admin:test_alert"),
        ]
    ])
    return text, kb


# =====================================================================
# 📌 ОБРАБОТЧИКИ КОМАНД И КНОПОК МЕНЮ
# =====================================================================
@dp.message(CommandStart())
async def cmd_start(message: Message):
    model, chat_id = ensure_user(message.from_user.id)
    await message.answer(
        f"👋 <b>Привет, {message.from_user.first_name}!</b>\n\n"
        f"Я универсальный бот на базе <b>Google Gemini AI</b>.\n"
        f"🔹 Принимаю <b>любые форматы</b>: текст, код, фото, видео, голосовые, кружочки, аудио, PDF и документы.\n"
        f"🔹 Выгружаю ответы текстом, кодом, файлами и изображениями.\n"
        f"🔹 Поддерживаю переключение между прошлыми и новыми чатами.\n\n"
        f"🤖 Текущая модель: <code>{model}</code>\n"
        f"💬 Активный чат: <code>#{chat_id}</code>",
        parse_mode="HTML",
        reply_markup=get_main_keyboard(message.from_user.id)
    )


@dp.message(Command("admin"))
@dp.message(F.text == "📊 Админ")
async def cmd_admin(message: Message):
    if message.from_user.id not in ADMIN_IDS:
        await message.answer("⛔ У вас нет прав администратора.")
        return
    text, kb = await render_admin_dashboard()
    await message.answer(text, parse_mode="HTML", reply_markup=kb)


@dp.callback_query(F.data.startswith("admin:"))
async def cb_admin(call: CallbackQuery):
    if call.from_user.id not in ADMIN_IDS:
        await call.answer("Нет доступа", show_alert=True)
        return

    action = call.data.split(":")[1]
    if action == "refresh":
        text, kb = await render_admin_dashboard()
        try:
            await call.message.edit_text(text, parse_mode="HTML", reply_markup=kb)
        except Exception:
            pass
        await call.answer("Данные обновлены!")
    elif action == "toggle_alerts":
        HEALTH_STATE["alerts_enabled"] = not HEALTH_STATE["alerts_enabled"]
        text, kb = await render_admin_dashboard()
        await call.message.edit_text(text, parse_mode="HTML", reply_markup=kb)
        await call.answer("Настройка уведомлений изменена")
    elif action == "test_alert":
        await call.message.answer(
            "🚨 <b>ТЕСТОВАЯ ТРЕВОГА: СЕРВЕРУ БО-БО!</b> 🚨\n"
            "Система оповещений раз в 10 минут исправна.",
            parse_mode="HTML"
        )
        await call.answer("Тестовый алерт отправлен!")


# --- ВЫБОР МОДЕЛИ ---
@dp.message(Command("models"))
@dp.message(F.text == "🤖 Выбор модели")
async def cmd_models(message: Message):
    model, _ = ensure_user(message.from_user.id)
    kb = await build_models_keyboard(message.from_user.id, page=0)
    await message.answer(
        f"🤖 <b>Выберите модель ИИ</b> (отсортированы от самых новых к старым):\n"
        f"Текущая: <code>{model}</code>",
        parse_mode="HTML",
        reply_markup=kb
    )


@dp.callback_query(F.data.startswith("modelpage:"))
async def cb_model_page(call: CallbackQuery):
    arg = call.data.split(":")[1]
    if arg == "refresh":
        await fetch_sorted_models(force=True)
        page = 0
        await call.answer("Список моделей обновлен из Google API!")
    else:
        page = int(arg)

    model, _ = ensure_user(call.from_user.id)
    kb = await build_models_keyboard(call.from_user.id, page=page)
    await call.message.edit_text(
        f"🤖 <b>Выберите модель ИИ</b> (от новых к старым):\nТекущая: <code>{model}</code>",
        parse_mode="HTML",
        reply_markup=kb
    )


@dp.callback_query(F.data.startswith("setmodel:"))
async def cb_set_model(call: CallbackQuery):
    _, new_model, page_str = call.data.split(":")
    set_user_model(call.from_user.id, new_model)
    kb = await build_models_keyboard(call.from_user.id, page=int(page_str))
    await call.message.edit_text(
        f"✅ <b>Модель успешно изменена!</b>\nТекущая: <code>{new_model}</code>",
        parse_mode="HTML",
        reply_markup=kb
    )
    await call.answer(f"Выбрана {new_model}")


# --- НАВИГАЦИЯ ПО ЧАТАМ (НОВЫЙ / ПРОШЛЫЙ / СЛЕДУЮЩИЙ / СПИСОК) ---
@dp.message(F.text == "➕ Новый чат")
@dp.message(Command("new"))
async def cmd_new_chat(message: Message):
    ensure_user(message.from_user.id)
    chat_id = create_new_chat(message.from_user.id)
    await message.answer(
        f"✨ <b>Создан новый чат #{chat_id}!</b>\nОтправьте сообщение, код или любой файл.",
        parse_mode="HTML"
    )


async def switch_adjacent_chat(message: Message, direction: int):
    """Переключает на прошлый (-1) или следующий (+1) чат."""
    _, active_chat_id = ensure_user(message.from_user.id)
    chats = get_user_chats(message.from_user.id)
    chat_ids = [c[0] for c in chats]

    if active_chat_id not in chat_ids:
        return

    idx = chat_ids.index(active_chat_id)
    target_idx = idx + direction

    if target_idx < 0:
        await message.answer("⬅️ Вы уже находитесь в самом первом (старом) чате.")
        return
    if target_idx >= len(chat_ids):
        await message.answer("➡️ Вы уже в самом последнем чате. Нажмите «➕ Новый чат», чтобы создать новый.")
        return

    new_chat_id, title, created_at = chats[target_idx]
    set_active_chat(message.from_user.id, new_chat_id)
    msgs = get_chat_messages(new_chat_id)

    preview = ""
    if msgs:
        last_msgs = msgs[-2:]
        preview_lines = []
        for r, txt, _, _, _ in last_msgs:
            who = "👤 Вы" if r == "user" else "🤖 ИИ"
            short = (txt or "[Файл]").replace("\n", " ")[:80]
            preview_lines.append(f"{who}: <i>{short}...</i>")
        preview = "\n\n<b>Последние сообщения:</b>\n" + "\n".join(preview_lines)

    await message.answer(
        f"📂 <b>Переключено на чат #{new_chat_id}</b> ({target_idx + 1} из {len(chats)})\n"
        f"📌 Тема: <b>{title}</b>\n"
        f"🕒 Создан: <code>{created_at}</code>"
        f"{preview}",
        parse_mode="HTML"
    )


@dp.message(F.text == "⬅️ Прошлый чат")
async def cmd_prev_chat(message: Message):
    await switch_adjacent_chat(message, direction=-1)


@dp.message(F.text == "➡️ Следующий чат")
async def cmd_next_chat(message: Message):
    await switch_adjacent_chat(message, direction=1)


@dp.message(F.text == "🗂 Все чаты")
@dp.message(Command("chats"))
async def cmd_all_chats(message: Message):
    _, active_chat_id = ensure_user(message.from_user.id)
    chats = get_user_chats(message.from_user.id)

    rows = []
    for c_id, title, _ in reversed(chats[-15:]):  # Показываем последние 15 чатов
        mark = "🟢 " if c_id == active_chat_id else "💬 "
        rows.append([InlineKeyboardButton(text=f"{mark}#{c_id}: {title}", callback_data=f"selectchat:{c_id}")])

    rows.append([InlineKeyboardButton(text="➕ Создать новый чат", callback_data="selectchat:new")])
    await message.answer(
        "🗂 <b>Ваши диалоги</b> (нажмите для переключения):",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=rows)
    )


@dp.callback_query(F.data.startswith("selectchat:"))
async def cb_select_chat(call: CallbackQuery):
    arg = call.data.split(":")[1]
    if arg == "new":
        chat_id = create_new_chat(call.from_user.id)
        await call.message.edit_text(f"✨ <b>Создан и выбран новый чат #{chat_id}!</b>", parse_mode="HTML")
        return

    chat_id = int(arg)
    set_active_chat(call.from_user.id, chat_id)
    msgs = get_chat_messages(chat_id)
    await call.message.edit_text(
        f"✅ <b>Активен чат #{chat_id}</b>\nСообщений в истории: <b>{len(msgs)}</b>",
        parse_mode="HTML"
    )
    await call.answer(f"Выбран чат #{chat_id}")


# =====================================================================
# 📥 ЭКСПОРТ ОТВЕТОВ ВО ВСЕ ФОРМАТЫ (.TXT, .MD, ФАЙЛЫ С КОДОМ)
# =====================================================================
@dp.callback_query(F.data.startswith("export:"))
async def cb_export_response(call: CallbackQuery):
    _, fmt, msg_id_str = call.data.split(":")
    text = get_message_by_id(int(msg_id_str))
    if not text:
        await call.answer("Текст сообщения не найден в БД", show_alert=True)
        return

    if fmt in ("txt", "md"):
        file_bytes = text.encode("utf-8")
        doc = BufferedInputFile(file_bytes, filename=f"gemini_response_{msg_id_str}.{fmt}")
        await call.message.answer_document(doc, caption=f"📄 Ответ сохранен в формате .{fmt}")
        await call.answer()

    elif fmt == "code":
        pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
        matches = pattern.findall(text)
        if not matches:
            await call.answer("Блоки кода не найдены", show_alert=True)
            return

        for idx, (lang, code_content) in enumerate(matches, start=1):
            ext = LANG_EXT_MAP.get(lang.lower().strip(), "txt") if lang else "txt"
            filename = f"code_{msg_id_str}_{idx}.{ext}"
            doc = BufferedInputFile(code_content.strip().encode("utf-8"), filename=filename)
            await call.message.answer_document(doc, caption=f"💻 Блок кода #{idx} ({lang or 'text'})")
        await call.answer("Код выгружен файлами!")


# =====================================================================
# 📎 ОБРАБОТКА ВСЕХ ТИПОВ ФАЙЛОВ И СООБЩЕНИЙ
# =====================================================================
async def extract_telegram_Attachment(message: Message) -> tuple[str | None, str | None, str | None, str | None]:
    """
    Извлекает любой медиафайл или документ из сообщения Telegram.
    Возвращает: (file_id, file_name, mime_type, local_ext)
    """
    if message.photo:
        ph = message.photo[-1]
        return ph.file_id, "photo.jpg", "image/jpeg", ".jpg"
    if message.video:
        v = message.video
        return v.file_id, v.file_name or "video.mp4", v.mime_type or "video/mp4", ".mp4"
    if message.video_note:
        vn = message.video_note
        return vn.file_id, "video_note.mp4", "video/mp4", ".mp4"
    if message.voice:
        vc = message.voice
        return vc.file_id, "voice.ogg", vc.mime_type or "audio/ogg", ".ogg"
    if message.audio:
        au = message.audio
        ext = Path(au.file_name).suffix if au.file_name else ".mp3"
        return au.file_id, au.file_name or f"audio{ext}", au.mime_type or "audio/mpeg", ext
    if message.sticker:
        st = message.sticker
        if st.is_video:
            return st.file_id, "sticker.webm", "video/webm", ".webm"
        return st.file_id, "sticker.webp", "image/webp", ".webp"
    if message.document:
        doc = message.document
        ext = Path(doc.file_name).suffix.lower() if doc.file_name else ""
        return doc.file_id, doc.file_name or "document", doc.mime_type or "application/octet-stream", ext

    return None, None, None, None


async def send_smart_response(message: Message, response, db_msg_id: int):
    """Отправляет все типы данных из ответа Gemini: текст, изображения, аудио, файлы."""
    # Проверяем наличие встроенных медиафайлов (картинок/аудио) в ответе модели
    if response.candidates:
        for cand in response.candidates:
            if not cand.content or not cand.content.parts:
                continue
            for part in cand.content.parts:
                if part.inline_data and part.inline_data.data:
                    mime = part.inline_data.mime_type or "application/octet-stream"
                    raw_bytes = part.inline_data.data
                    if mime.startswith("image/"):
                        ext = mime.split("/")[-1]
                        await message.answer_photo(
                            BufferedInputFile(raw_bytes, filename=f"generated.{ext}"),
                            caption="🖼 Сгенерированное изображение"
                        )
                    elif mime.startswith("audio/"):
                        ext = mime.split("/")[-1]
                        await message.answer_audio(
                            BufferedInputFile(raw_bytes, filename=f"audio.{ext}"),
                            caption="🔊 Аудио от Gemini"
                        )
                    else:
                        await message.answer_document(
                            BufferedInputFile(raw_bytes, filename="output.bin")
                        )

    text = ""
    try:
        text = response.text or ""
    except Exception:
        pass

    if not text:
        return

    has_code = "```" in text
    kb = get_response_export_kb(db_msg_id, has_code)

    # Разбиваем длинный текст на части по 4000 символов (лимит Telegram — 4096)
    chunk_size = 4000
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    for i, chunk in enumerate(chunks):
        reply_kb = kb if i == len(chunks) - 1 else None
        try:
            await message.answer(chunk, parse_mode="Markdown", reply_markup=reply_kb)
        except Exception:
            # Если Markdown невалидный из-за спецсимволов в коде, отправляем чистым текстом
            await message.answer(chunk, reply_markup=reply_kb)

    # Если ответ огромный (> 12 000 символов), дополнительно прикладываем полным файлом
    if len(text) > 12000:
        doc = BufferedInputFile(text.encode("utf-8"), filename=f"full_response_{db_msg_id}.md")
        await message.answer_document(doc, caption="📚 Полный текст ответа в файле")


@dp.message()
async def handle_universal_message(message: Message):
    user_id = message.from_user.id
    model_name, chat_id = ensure_user(user_id)

    user_prompt = message.text or message.caption or ""
    file_id, file_name, mime_type, ext = await extract_telegram_Attachment(message)

    if not user_prompt and not file_id:
        await message.answer("⚠️ Пожалуйста, отправьте текст, код или любой медиафайл.")
        return

    status_msg = await message.answer(f"⏳ <i>Обрабатываю запрос ({model_name})...</i>", parse_mode="HTML")
    update_stats(req=1)

    uploaded_gemini_file = None
    local_temp_path = None

    try:
        # 1. Если модель семейства Imagen — генерируем картинку
        if model_name.startswith("imagen"):
            img_res = await client.aio.models.generate_images(
                model=model_name,
                prompt=user_prompt or "Beautiful futuristic artwork",
                config=genai_types.GenerateImagesConfig(number_of_images=1)
            )
            for gen_img in img_res.generated_images:
                img_bytes = gen_img.image.image_bytes
                await message.answer_photo(
                    BufferedInputFile(img_bytes, filename="imagen.png"),
                    caption=f"🎨 Сгенерировано моделью <code>{model_name}</code>",
                    parse_mode="HTML"
                )
            update_stats(success=1)
            await status_msg.delete()
            return

        # 2. Если прикреплен файл (фото, видео, аудио, кружочек, документ, код)
        if file_id:
            update_stats(files=1)
            await status_msg.edit_text(f"📥 <i>Скачиваю файл {file_name}...</i>", parse_mode="HTML")

            tg_file = await bot.get_file(file_id)
            fd, local_temp_path = tempfile.mkstemp(suffix=ext or ".bin")
            os.close(fd)
            await bot.download_file(tg_file.file_path, destination=local_temp_path)

            # Проверяем: если это текстовый/кодовый файл — читаем его напрямую в промпт
            if ext in TEXT_CODE_EXTENSIONS or (mime_type and mime_type.startswith("text/")):
                with open(local_temp_path, "r", encoding="utf-8", errors="replace") as f:
                    code_text = f.read()
                user_prompt = (
                    f"Содержимое файла `{file_name}`:\n```\n{code_text}\n```\n\n"
                    f"{user_prompt or 'Проанализируй этот файл/код.'}"
                )
            else:
                # Иначе загружаем через Gemini File API (поддерживает видео, аудио, PDF, фото)
                await status_msg.edit_text(
                    f"☁️ <i>Загружаю {file_name} в Google Gemini AI...</i>",
                    parse_mode="HTML"
                )
                uploaded_gemini_file = await client.aio.files.upload(
                    file=local_temp_path,
                    config={"mime_type": mime_type} if mime_type and mime_type != "application/octet-stream" else None
                )

                # Ждем обработки файла (обязательно для видео и больших аудио)
                while uploaded_gemini_file.state.name == "PROCESSING":
                    await status_msg.edit_text(
                        "🎬 <i>Google AI обрабатывает медиафайл, подождите пару секунд...</i>",
                        parse_mode="HTML"
                    )
                    await asyncio.sleep(3)
                    uploaded_gemini_file = await client.aio.files.get(name=uploaded_gemini_file.name)

                if uploaded_gemini_file.state.name == "FAILED":
                    raise RuntimeError("Не удалось обработать медиафайл на стороне Google AI.")

                if not user_prompt:
                    user_prompt = "Подробно проанализируй и опиши этот файл."

        # 3. Сохраняем сообщение пользователя в БД
        add_message(
            chat_id=chat_id,
            role="user",
            text=user_prompt,
            file_uri=uploaded_gemini_file.uri if uploaded_gemini_file else None,
            file_mime=uploaded_gemini_file.mime_type if uploaded_gemini_file else None,
        )

        # 4. Собираем историю текущего чата для передачи контекста
        db_history = get_chat_messages(chat_id)
        contents = []
        now_ts = time.time()

        for role, txt, f_uri, f_mime, created_ts in db_history:
            parts = []
            # Файлы в Gemini File API живут 48 часов (172800 сек)
            if f_uri and f_mime and (now_ts - created_ts) < 170000:
                parts.append(genai_types.Part.from_uri(file_uri=f_uri, mime_type=f_mime))
            if txt:
                parts.append(genai_types.Part.from_text(text=txt))
            if parts:
                contents.append(genai_types.Content(role=role, parts=parts))

        await status_msg.edit_text(f"🧠 <i>{model_name} думает...</i>", parse_mode="HTML")

        # 5. Запрос к Gemini API
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=contents,
        )

        ans_text = ""
        try:
            ans_text = response.text or "[Медиа-ответ]"
        except Exception:
            ans_text = "[Медиа-ответ]"

        db_msg_id = add_message(chat_id=chat_id, role="model", text=ans_text)
        update_stats(success=1)

        await status_msg.delete()
        await send_smart_response(message, response, db_msg_id)

    except Exception as e:
        update_stats(err=1)
        logger.exception("Ошибка при обработке сообщения:")
        await status_msg.edit_text(
            f"❌ <b>Произошла ошибка:</b>\n<code>{type(e).__name__}: {str(e)[:300]}</code>\n\n"
            f"💡 Попробуйте сменить модель через кнопку «🤖 Выбор модели» или создать «➕ Новый чат».",
            parse_mode="HTML"
        )
    finally:
        if local_temp_path and os.path.exists(local_temp_path):
            try:
                os.remove(local_temp_path)
            except OSError:
                pass


# =====================================================================
# 🚀 ЗАПУСК БОТА
# =====================================================================
async def main():
    init_db()
    logger.info("Кэширование списка моделей...")
    await fetch_sorted_models(force=True)

    # Запускаем фоновый мониторинг (раз в 10 минут)
    asyncio.create_task(background_health_monitor())

    logger.info("Бот запущен!")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
