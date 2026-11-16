import asyncio
import datetime
import gc
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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =====================================================================
# ⚙️ НАСТРОЙКИ
# =====================================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "ВАШ_ТОКЕН_ТЕЛЕГРАМ_БОТА")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "ВАШ_КЛЮЧ_GEMINI_API")
ADMIN_IDS = [123456789]  # Ваш Telegram ID

DEFAULT_MODEL = "gemini-4-hydrogen"
DB_PATH = "gemini_bot.db"

# Лимиты времени и памяти (1 CPU / 1 GB RAM)
HEALTH_CHECK_INTERVAL = 600          # Проверка сервера: каждые 10 минут
CLEANUP_INTERVAL = 1800              # Автоочистка: каждые 30 минут
CHAT_TTL_SECONDS = 7 * 86400         # Удаление всех диалогов: через 7 дней (1 неделю)
LARGE_FILE_TTL_SECONDS = 24 * 3600   # Удаление файлов > 10 МБ: через 24 часа
SMALL_FILE_TTL_SECONDS = 48 * 3600   # Удаление остальных файлов: через 48 часов
LARGE_FILE_BYTES = 10 * 1024 * 1024  # Порог тяжелого файла: 10 МБ

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()
client = genai.Client(api_key=GEMINI_API_KEY)

START_TIME = time.time()
MODELS_CACHE = []
HEALTH_STATE = {
    "is_healthy": True,
    "last_error": None,
    "last_ping_ms": 0,
    "alerts_enabled": True,
}

TEXT_CODE_EXTENSIONS = {
    ".txt", ".py", ".js", ".ts", ".html", ".css", ".json", ".xml", ".yaml", ".yml",
    ".md", ".csv", ".sql", ".sh", ".bat", ".ps1", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".java", ".go", ".rs", ".php", ".rb", ".kt", ".swift", ".ini", ".cfg",
    ".env", ".dockerfile", ".toml", ".log"
}

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
# 🗄 БАЗА ДАННЫХ (ОПТИМИЗИРОВАНА ПОД 1 ГБ ОЗУ, ИСТОРИЯ ТОЛЬКО В ТЕКСТЕ)
# =====================================================================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=FILE;")
    conn.execute("PRAGMA cache_size=-4000;")  # Не более 4 МБ ОЗУ под кэш БД
    return conn


def init_db():
    with get_db() as conn:
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
                created_at REAL
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
                file_api_name TEXT,
                file_size INTEGER DEFAULT 0,
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
    now = time.time()
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT current_model, active_chat_id FROM users WHERE user_id = ?", (user_id,))
        row = c.fetchone()
        if not row:
            c.execute("INSERT INTO chats (user_id, title, created_at) VALUES (?, ?, ?)", (user_id, "Новый диалог", now))
            chat_id = c.lastrowid
            c.execute("INSERT INTO users (user_id, current_model, active_chat_id) VALUES (?, ?, ?)", (user_id, DEFAULT_MODEL, chat_id))
            conn.commit()
            return DEFAULT_MODEL, chat_id

        model, chat_id = row
        c.execute("SELECT chat_id FROM chats WHERE chat_id = ? AND user_id = ?", (chat_id, user_id))
        if not c.fetchone():
            c.execute("INSERT INTO chats (user_id, title, created_at) VALUES (?, ?, ?)", (user_id, "Новый диалог", now))
            chat_id = c.lastrowid
            c.execute("UPDATE users SET active_chat_id = ? WHERE user_id = ?", (chat_id, user_id))
            conn.commit()
        return model, chat_id


def create_new_chat(user_id: int) -> int:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("INSERT INTO chats (user_id, title, created_at) VALUES (?, ?, ?)", (user_id, "Новый диалог", time.time()))
        chat_id = c.lastrowid
        c.execute("UPDATE users SET active_chat_id = ? WHERE user_id = ?", (chat_id, user_id))
        conn.commit()
        return chat_id


def get_user_chats(user_id: int):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT chat_id, title, created_at FROM chats WHERE user_id = ? ORDER BY chat_id ASC", (user_id,))
        return c.fetchall()


def set_active_chat(user_id: int, chat_id: int):
    with get_db() as conn:
        conn.execute("UPDATE users SET active_chat_id = ? WHERE user_id = ?", (chat_id, user_id))
        conn.commit()


def set_user_model(user_id: int, model_name: str):
    with get_db() as conn:
        conn.execute("UPDATE users SET current_model = ? WHERE user_id = ?", (model_name, user_id))
        conn.commit()


def add_message(chat_id: int, role: str, text: str, file_uri: str = None, file_mime: str = None, file_api_name: str = None, file_size: int = 0) -> int:
    with get_db() as conn:
        c = conn.cursor()
        c.execute(
            """INSERT INTO messages (chat_id, role, text, file_uri, file_mime, file_api_name, file_size, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (chat_id, role, text, file_uri, file_mime, file_api_name, file_size, time.time())
        )
        msg_id = c.lastrowid
        if role == "user" and text:
            c.execute("SELECT COUNT(*) FROM messages WHERE chat_id = ?", (chat_id,))
            if c.fetchone()[0] <= 1:
                clean_title = text.strip().replace("\n", " ")[:35]
                c.execute("UPDATE chats SET title = ? WHERE chat_id = ?", (clean_title, chat_id))
        conn.commit()
        return msg_id


def get_chat_messages(chat_id: int):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT role, text, file_uri, file_mime, file_size, created_at FROM messages WHERE chat_id = ? ORDER BY msg_id ASC", (chat_id,))
        return c.fetchall()


def get_message_by_id(msg_id: int):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT text FROM messages WHERE msg_id = ?", (msg_id,))
        row = c.fetchone()
        return row[0] if row else None


def update_stats(req=0, success=0, err=0, files=0):
    with get_db() as conn:
        conn.execute("""
            UPDATE stats SET
                total_requests = total_requests + ?,
                success_requests = success_requests + ?,
                error_requests = error_requests + ?,
                files_processed = files_processed + ?
            WHERE id = 1
        """, (req, success, err, files))
        conn.commit()


def get_stats():
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT total_requests, success_requests, error_requests, files_processed FROM stats WHERE id = 1")
        st = c.fetchone()
        c.execute("SELECT COUNT(*) FROM users")
        u_cnt = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM chats")
        c_cnt = c.fetchone()[0]
        return {"total": st[0], "success": st[1], "errors": st[2], "files": st[3], "users": u_cnt, "chats": c_cnt}


# =====================================================================
# 🧹 АВТООЧИСТКА (ФАЙЛЫ > 10 МБ ЧЕРЕЗ 24Ч, ДИАЛОГИ ЧЕРЕЗ 7 ДНЕЙ)
# =====================================================================
async def run_cleanup():
    now = time.time()
    large_cutoff = now - LARGE_FILE_TTL_SECONDS
    small_cutoff = now - SMALL_FILE_TTL_SECONDS
    chat_cutoff = now - CHAT_TTL_SECONDS

    api_files_to_delete = []
    with get_db() as conn:
        c = conn.cursor()
        # 1. Находим файлы > 10 МБ старше 24ч, остальные медиа старше 48ч и файлы из удаляемых чатов (> 7 дней)
        c.execute("""
            SELECT msg_id, file_api_name FROM messages
            WHERE file_uri IS NOT NULL AND (
                (file_size >= ? AND created_at < ?) OR
                (created_at < ?) OR
                (chat_id IN (SELECT chat_id FROM chats WHERE created_at < ?))
            )
        """, (LARGE_FILE_BYTES, large_cutoff, small_cutoff, chat_cutoff))
        for msg_id, f_name in c.fetchall():
            if f_name:
                api_files_to_delete.append(f_name)
            c.execute("UPDATE messages SET file_uri=NULL, file_mime=NULL, file_api_name=NULL WHERE msg_id=?", (msg_id,))

        # 2. Удаляем все диалоги и сообщения старше 7 дней
        c.execute("DELETE FROM chats WHERE created_at < ?", (chat_cutoff,))
        c.execute("DELETE FROM messages WHERE created_at < ? OR chat_id NOT IN (SELECT chat_id FROM chats)", (chat_cutoff,))
        conn.commit()

    for f_name in set(api_files_to_delete):
        try:
            await client.aio.files.delete(name=f_name)
        except Exception:
            pass

    with get_db() as conn:
        conn.execute("VACUUM")
    gc.collect()


async def background_cleanup_loop():
    while True:
        try:
            await run_cleanup()
        except Exception as e:
            logger.error(f"Ошибка автоочистки: {e}")
        await asyncio.sleep(CLEANUP_INTERVAL)


# =====================================================================
# 🎹 КЛАВИАТУРЫ И СПИСОК МОДЕЛЕЙ
# =====================================================================
def get_main_keyboard(user_id: int) -> ReplyKeyboardMarkup:
    kb = [
        [KeyboardButton(text="⬅️ Прошлый чат"), KeyboardButton(text="➕ Новый чат"), KeyboardButton(text="➡️ Следующий чат")],
        [KeyboardButton(text="🗂 Все чаты"), KeyboardButton(text="🤖 Выбор модели")]
    ]
    if user_id in ADMIN_IDS:
        kb[1].append(KeyboardButton(text="📊 Админ"))
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)


def get_response_export_kb(db_msg_id: int, has_code: bool) -> InlineKeyboardMarkup:
    rows = [[
        InlineKeyboardButton(text="📄 Скачать .txt", callback_data=f"export:txt:{db_msg_id}"),
        InlineKeyboardButton(text="📝 Скачать .md", callback_data=f"export:md:{db_msg_id}"),
    ]]
    if has_code:
        rows.append([InlineKeyboardButton(text="💻 Выгрузить код файлом", callback_data=f"export:code:{db_msg_id}")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def parse_model_sort_key(model_name: str):
    ver_match = re.search(r"(\d+)\.(\d+)", model_name)
    if ver_match:
        major, minor = int(ver_match.group(1)), int(ver_match.group(2))
    else:
        single_ver = re.search(r"-(\d+)-", model_name)
        major = int(single_ver.group(1)) if single_ver else 0
        minor = 0

    tier = 1
    if "carbon" in model_name: tier = 6
    elif "argon" in model_name: tier = 5
    elif "hydrogen" in model_name: tier = 4
    elif "pro" in model_name: tier = 3
    elif "flash" in model_name and "lite" not in model_name: tier = 2

    is_latest = 1 if ("latest" in model_name or "preview" in model_name or "exp" in model_name) else 0
    return (major, minor, tier, is_latest, model_name)


async def fetch_sorted_models(force=False) -> list[str]:
    global MODELS_CACHE
    if MODELS_CACHE and not force:
        return MODELS_CACHE

    fallback = [
        "gemini-4-carbon", "gemini-4-argon", "gemini-4-hydrogen",
        "gemini-3.8-flash", "gemini-3.5-flash-lite",
        "gemini-2.5-pro", "gemini-2.5-flash", "imagen-3.0-generate-002",
    ]
    try:
        raw_models = set(fallback)
        pager = await client.aio.models.list()
        async for m in pager:
            name = m.name.replace("models/", "")
            if any(x in name for x in ["embedding", "aqa", "bison", "gecko", "learnlm"]):
                continue
            if name.startswith("gemini") or name.startswith("imagen"):
                raw_models.add(name)
        MODELS_CACHE = sorted(list(raw_models), key=parse_model_sort_key, reverse=True)
        return MODELS_CACHE
    except Exception:
        return sorted(fallback, key=parse_model_sort_key, reverse=True)


async def build_models_keyboard(user_id: int, page: int = 0) -> InlineKeyboardMarkup:
    models = await fetch_sorted_models()
    current_model, _ = ensure_user(user_id)
    per_page = 7
    total_pages = max(1, (len(models) + per_page - 1) // per_page)
    page = max(0, min(page, total_pages - 1))
    page_models = models[page * per_page : (page + 1) * per_page]

    rows = []
    for idx, m in enumerate(page_models):
        prefix = "✅ " if m == current_model else ""
        badge = "🔥 " if (page * per_page + idx) < 3 else ""
        rows.append([InlineKeyboardButton(text=f"{prefix}{badge}{m}", callback_data=f"setmodel:{m}:{page}")])

    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton(text="⬅️", callback_data=f"modelpage:{page - 1}"))
    nav.append(InlineKeyboardButton(text=f"{page + 1}/{total_pages}", callback_data="noop"))
    if page < total_pages - 1:
        nav.append(InlineKeyboardButton(text="➡️", callback_data=f"modelpage:{page + 1}"))
    rows.append(nav)
    rows.append([InlineKeyboardButton(text="🔄 Обновить из API", callback_data="modelpage:refresh")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


# =====================================================================
# 🚨 МОНИТОРИНГ СЕРВЕРА (КАЖДЫЕ 10 МИНУТ) И /admin
# =====================================================================
async def check_service_health() -> tuple[bool, str, int]:
    t0 = time.monotonic()
    try:
        await asyncio.wait_for(client.aio.models.list(), timeout=15.0)
        ping_ms = int((time.monotonic() - t0) * 1000)
        cpu = psutil.cpu_percent(interval=0.4)
        ram = psutil.virtual_memory().percent
        if cpu > 92 or ram > 92:
            return False, f"Перегрузка VPS (1GB)! CPU: {cpu}%, RAM: {ram}%", ping_ms
        return True, "OK", ping_ms
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}", int((time.monotonic() - t0) * 1000)


async def background_health_monitor():
    while True:
        await asyncio.sleep(HEALTH_CHECK_INTERVAL)
        ok, details, ping_ms = await check_service_health()
        now_str = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        HEALTH_STATE.update({"is_healthy": ok, "last_ping_ms": ping_ms})
        if not ok:
            HEALTH_STATE["last_error"] = f"[{now_str}] {details}"
            if HEALTH_STATE["alerts_enabled"]:
                msg = (
                    f"🚨 <b>СЕРВЕРУ БО-БО!</b>\n🕒 {now_str}\n"
                    f"⚠️ Причина: <code>{details}</code>\n"
                    f"💻 CPU: {psutil.cpu_percent()}% | RAM: {psutil.virtual_memory().percent}%"
                )
                for adm in ADMIN_IDS:
                    try:
                        await bot.send_message(adm, msg, parse_mode="HTML")
                    except Exception:
                        pass


async def render_admin_dashboard() -> tuple[str, InlineKeyboardMarkup]:
    ok, details, ping_ms = await check_service_health()
    st = get_stats()
    td = datetime.timedelta(seconds=int(time.time() - START_TIME))
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    status_str = "🟢 OK" if ok else f"🔴 БО-БО ({details})"

    text = (
        f"📊 <b>АДМИН-ПАНЕЛЬ (1 CPU / 1 GB RAM)</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"• Статус: <b>{status_str}</b>\n"
        f"• Пинг Gemini API: <code>{ping_ms} мс</code>\n"
        f"• Рантайм: <code>{td}</code>\n"
        f"• Запросы: <b>{st['total']}</b> (OK: {st['success']} | Err: {st['errors']})\n"
        f"• Файлов прокачано: <b>{st['files']}</b> | Чатов (7д): <b>{st['chats']}</b>\n\n"
        f"🖥 <b>Ресурсы VPS:</b>\n"
        f"• CPU: <code>{psutil.cpu_percent()}%</code>\n"
        f"• RAM: <code>{mem.percent}% ({mem.used // (1024**2)}/{mem.total // (1024**2)} МБ)</code>\n"
        f"• Диск: <code>{disk.percent}%</code>"
    )
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔄 Обновить", callback_data="admin:refresh"),
         InlineKeyboardButton(text="🧹 Очистить старое", callback_data="admin:cleanup")],
        [InlineKeyboardButton(text="🔔 Вкл/Выкл алерты", callback_data="admin:toggle_alerts"),
         InlineKeyboardButton(text="🚨 Тест «Бо-бо»", callback_data="admin:test_alert")]
    ])
    return text, kb


# =====================================================================
# 📌 ОБРАБОТЧИКИ КОМАНД
# =====================================================================
@dp.message(CommandStart())
async def cmd_start(message: Message):
    model, chat_id = ensure_user(message.from_user.id)
    await message.answer(
        f"👋 <b>Привет!</b>\n🤖 Модель: <code>{model}</code> | Чат: <code>#{chat_id}</code>\n"
        f"<i>История хранится в тексте 7 дней. Файлы > 10 МБ удаляются через 24 часа.</i>",
        parse_mode="HTML", reply_markup=get_main_keyboard(message.from_user.id)
    )


@dp.message(Command("admin"))
@dp.message(F.text == "📊 Админ")
async def cmd_admin(message: Message):
    if message.from_user.id in ADMIN_IDS:
        text, kb = await render_admin_dashboard()
        await message.answer(text, parse_mode="HTML", reply_markup=kb)


@dp.callback_query(F.data.startswith("admin:"))
async def cb_admin(call: CallbackQuery):
    if call.from_user.id not in ADMIN_IDS:
        return
    act = call.data.split(":")[1]
    if act == "cleanup":
        await run_cleanup()
        await call.answer("Очистка завершена, ОЗУ освобождено!")
    elif act == "toggle_alerts":
        HEALTH_STATE["alerts_enabled"] = not HEALTH_STATE["alerts_enabled"]
        await call.answer(f"Алерты: {HEALTH_STATE['alerts_enabled']}")
    elif act == "test_alert":
        await call.message.answer("🚨 <b>ТЕСТ: СЕРВЕРУ БО-БО!</b>", parse_mode="HTML")
        await call.answer()
    text, kb = await render_admin_dashboard()
    try:
        await call.message.edit_text(text, parse_mode="HTML", reply_markup=kb)
    except Exception:
        pass


@dp.message(Command("models"))
@dp.message(F.text == "🤖 Выбор модели")
async def cmd_models(message: Message):
    model, _ = ensure_user(message.from_user.id)
    kb = await build_models_keyboard(message.from_user.id, 0)
    await message.answer(f"🤖 <b>Текущая модель:</b> <code>{model}</code>", parse_mode="HTML", reply_markup=kb)


@dp.callback_query(F.data.startswith("modelpage:"))
async def cb_model_page(call: CallbackQuery):
    arg = call.data.split(":")[1]
    if arg == "refresh":
        await fetch_sorted_models(force=True)
        page = 0
    else:
        page = int(arg)
    model, _ = ensure_user(call.from_user.id)
    kb = await build_models_keyboard(call.from_user.id, page)
    await call.message.edit_text(f"🤖 <b>Текущая модель:</b> <code>{model}</code>", parse_mode="HTML", reply_markup=kb)


@dp.callback_query(F.data.startswith("setmodel:"))
async def cb_set_model(call: CallbackQuery):
    _, new_model, page_str = call.data.split(":")
    set_user_model(call.from_user.id, new_model)
    kb = await build_models_keyboard(call.from_user.id, int(page_str))
    await call.message.edit_text(f"✅ <b>Выбрана модель:</b> <code>{new_model}</code>", parse_mode="HTML", reply_markup=kb)


@dp.message(F.text == "➕ Новый чат")
@dp.message(Command("new"))
async def cmd_new_chat(message: Message):
    ensure_user(message.from_user.id)
    cid = create_new_chat(message.from_user.id)
    await message.answer(f"✨ <b>Создан новый чат #{cid}!</b>", parse_mode="HTML")


async def switch_chat(message: Message, delta: int):
    _, active_id = ensure_user(message.from_user.id)
    chats = get_user_chats(message.from_user.id)
    ids = [c[0] for c in chats]
    if active_id not in ids:
        return
    idx = ids.index(active_id) + delta
    if 0 <= idx < len(chats):
        cid, title, _ = chats[idx]
        set_active_chat(message.from_user.id, cid)
        await message.answer(f"📂 <b>Чат #{cid}:</b> {title}", parse_mode="HTML")
    else:
        await message.answer("⚠️ Крайний чат в списке.")


@dp.message(F.text == "⬅️ Прошлый чат")
async def cmd_prev(message: Message):
    await switch_chat(message, -1)


@dp.message(F.text == "➡️ Следующий чат")
async def cmd_next(message: Message):
    await switch_chat(message, 1)


@dp.message(F.text == "🗂 Все чаты")
async def cmd_all_chats(message: Message):
    _, active_id = ensure_user(message.from_user.id)
    chats = get_user_chats(message.from_user.id)
    rows = [[InlineKeyboardButton(text=f"{'🟢' if c[0]==active_id else '💬'} #{c[0]}: {c[1]}", callback_data=f"selchat:{c[0]}")] for c in reversed(chats[-15:])]
    rows.append([InlineKeyboardButton(text="➕ Новый чат", callback_data="selchat:new")])
    await message.answer("🗂 <b>Чаты за последние 7 дней:</b>", parse_mode="HTML", reply_markup=InlineKeyboardMarkup(inline_keyboard=rows))


@dp.callback_query(F.data.startswith("selchat:"))
async def cb_sel_chat(call: CallbackQuery):
    arg = call.data.split(":")[1]
    cid = create_new_chat(call.from_user.id) if arg == "new" else int(arg)
    set_active_chat(call.from_user.id, cid)
    await call.message.edit_text(f"✅ <b>Активен чат #{cid}</b>", parse_mode="HTML")


@dp.callback_query(F.data.startswith("export:"))
async def cb_export(call: CallbackQuery):
    _, fmt, mid = call.data.split(":")
    text = get_message_by_id(int(mid))
    if not text:
        return await call.answer("Сообщение уже удалено", show_alert=True)
    if fmt in ("txt", "md"):
        await call.message.answer_document(BufferedInputFile(text.encode("utf-8"), filename=f"msg_{mid}.{fmt}"))
    elif fmt == "code":
        for i, (lang, code) in enumerate(re.findall(r"```(\w+)?\n(.*?)```", text, re.DOTALL), 1):
            ext = LANG_EXT_MAP.get(lang.lower().strip(), "txt") if lang else "txt"
            await call.message.answer_document(BufferedInputFile(code.strip().encode("utf-8"), filename=f"code_{mid}_{i}.{ext}"))
    await call.answer()


# =====================================================================
# 📎 ПОТОКОВАЯ ОБРАБОТКА ВСЕХ ФАЙЛОВ И СООБЩЕНИЙ
# =====================================================================
async def extract_attachment(m: Message):
    if m.photo: return m.photo[-1].file_id, "photo.jpg", "image/jpeg", ".jpg", m.photo[-1].file_size or 0
    if m.video: return m.video.file_id, m.video.file_name or "video.mp4", m.video.mime_type or "video/mp4", ".mp4", m.video.file_size or 0
    if m.video_note: return m.video_note.file_id, "vnote.mp4", "video/mp4", ".mp4", m.video_note.file_size or 0
    if m.voice: return m.voice.file_id, "voice.ogg", m.voice.mime_type or "audio/ogg", ".ogg", m.voice.file_size or 0
    if m.audio:
        ext = Path(m.audio.file_name).suffix if m.audio.file_name else ".mp3"
        return m.audio.file_id, m.audio.file_name or f"audio{ext}", m.audio.mime_type or "audio/mpeg", ext, m.audio.file_size or 0
    if m.document:
        ext = Path(m.document.file_name).suffix.lower() if m.document.file_name else ""
        return m.document.file_id, m.document.file_name or "doc", m.document.mime_type or "application/octet-stream", ext, m.document.file_size or 0
    return None, None, None, None, 0


@dp.message()
async def handle_message(message: Message):
    user_id = message.from_user.id
    model_name, chat_id = ensure_user(user_id)
    prompt = message.text or message.caption or ""
    file_id, fname, mime, ext, fsize = await extract_attachment(message)

    if not prompt and not file_id:
        return

    status = await message.answer(f"⏳ <i>Обработка ({model_name})...</i>", parse_mode="HTML")
    update_stats(req=1)
    g_file = None
    tmp_path = None

    try:
        if model_name.startswith("imagen"):
            res = await client.aio.models.generate_images(
                model=model_name, prompt=prompt or "Art", config=genai_types.GenerateImagesConfig(number_of_images=1)
            )
            for img in res.generated_images:
                await message.answer_photo(BufferedInputFile(img.image.image_bytes, filename="imagen.png"))
            add_message(chat_id, "user", prompt)
            add_message(chat_id, "model", f"[Сгенерировано изображение: {prompt}]")
            update_stats(success=1)
            return await status.delete()

        if file_id:
            update_stats(files=1)
            tg_f = await bot.get_file(file_id)
            fd, tmp_path = tempfile.mkstemp(suffix=ext or ".bin")
            os.close(fd)
            await bot.download_file(tg_f.file_path, destination=tmp_path)
            fsize = os.path.getsize(tmp_path) or fsize

            if (ext in TEXT_CODE_EXTENSIONS or (mime and mime.startswith("text/"))) and fsize < 512 * 1024:
                with open(tmp_path, "r", encoding="utf-8", errors="replace") as f:
                    prompt = f"Файл `{fname}`:\n```\n{f.read()}\n```\n\n{prompt or 'Проанализируй.'}"
                os.remove(tmp_path)
                tmp_path = None
            else:
                await status.edit_text(f"☁️ <i>Прокачка {fname} в Google AI...</i>", parse_mode="HTML")
                g_file = await client.aio.files.upload(
                    file=tmp_path,
                    config={"mime_type": mime} if mime and mime != "application/octet-stream" else None
                )
                # Сразу стираем локальный файл с диска!
                os.remove(tmp_path)
                tmp_path = None

                while g_file.state.name == "PROCESSING":
                    await asyncio.sleep(2)
                    g_file = await client.aio.files.get(name=g_file.name)

                if not prompt:
                    prompt = f"[Файл: {fname}] Проанализируй содержимое."

        add_message(
            chat_id, "user", prompt,
            file_uri=g_file.uri if g_file else None,
            file_mime=g_file.mime_type if g_file else None,
            file_api_name=g_file.name if g_file else None,
            file_size=fsize
        )

        # Собираем историю чата
        now = time.time()
        contents = []
        for r_role, r_txt, f_uri, f_mime, f_sz, c_ts in get_chat_messages(chat_id):
            parts = []
            ttl = LARGE_FILE_TTL_SECONDS if (f_sz and f_sz >= LARGE_FILE_BYTES) else SMALL_FILE_TTL_SECONDS
            if f_uri and f_mime and (now - c_ts) < ttl:
                parts.append(genai_types.Part.from_uri(file_uri=f_uri, mime_type=f_mime))
            if r_txt:
                parts.append(genai_types.Part.from_text(text=r_txt))
            if parts:
                contents.append(genai_types.Content(role=r_role, parts=parts))

        response = await client.aio.models.generate_content(model=model_name, contents=contents)

        # Отправляем медиа из ответа, если они есть (не сохраняя их в БД)
        if response.candidates:
            for cand in response.candidates:
                if cand.content and cand.content.parts:
                    for part in cand.content.parts:
                        if part.inline_data and part.inline_data.data:
                            m_type = part.inline_data.mime_type or ""
                            if m_type.startswith("image/"):
                                await message.answer_photo(BufferedInputFile(part.inline_data.data, filename="out.png"))
                            elif m_type.startswith("audio/"):
                                await message.answer_audio(BufferedInputFile(part.inline_data.data, filename="out.mp3"))

        ans_text = ""
        try:
            ans_text = response.text or "[Медиа-ответ]"
        except Exception:
            ans_text = "[Медиа-ответ]"

        db_msg_id = add_message(chat_id, "model", ans_text)
        update_stats(success=1)
        await status.delete()

        kb = get_response_export_kb(db_msg_id, "```" in ans_text)
        chunks = [ans_text[i : i + 4000] for i in range(0, len(ans_text), 4000)]
        for i, ch in enumerate(chunks):
            r_kb = kb if i == len(chunks) - 1 else None
            try:
                await message.answer(ch, parse_mode="Markdown", reply_markup=r_kb)
            except Exception:
                await message.answer(ch, reply_markup=r_kb)

    except Exception as e:
        update_stats(err=1)
        await status.edit_text(f"❌ <b>Ошибка:</b> <code>{type(e).__name__}: {str(e)[:200]}</code>", parse_mode="HTML")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        gc.collect()


async def main():
    init_db()
    await fetch_sorted_models(force=True)
    asyncio.create_task(background_health_monitor())
    asyncio.create_task(background_cleanup_loop())
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
