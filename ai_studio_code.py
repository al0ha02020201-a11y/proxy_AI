import asyncio
import base64
from contextlib import asynccontextmanager
import gc
import json
import logging
import os
import re
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from google import genai
from google.genai import types as genai_types

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =====================================================================
# ⚙️ НАСТРОЙКИ (ПОД 1 ЯДРО И 1 ГБ ОЗУ)
# =====================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "ВАШ_КЛЮЧ_GEMINI_API")
DEFAULT_MODEL = "gemini-4-hydrogen"
DB_PATH = "gemini_web.db"
HOST = "0.0.0.0"
PORT = 8000

# Правила автоочистки и потоковой прокачки
CLEANUP_INTERVAL = 1800              # Проверка каждые 30 минут
CHAT_TTL_SECONDS = 7 * 86400         # Все диалоги удаляются через 7 дней (1 неделю)
LARGE_FILE_TTL_SECONDS = 24 * 3600   # Файлы > 10 МБ удаляются через 24 часа
SMALL_FILE_TTL_SECONDS = 48 * 3600   # Остальные медиа удаляются через 48 часов
LARGE_FILE_BYTES = 10 * 1024 * 1024  # Порог 10 МБ
CHUNK_SIZE = 1024 * 1024             # Чтение файлов кусочками по 1 МБ мимо ОЗУ

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

client = genai.Client(api_key=GEMINI_API_KEY)
MODELS_CACHE = []

TEXT_CODE_EXTENSIONS = {
    ".txt", ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".json", ".xml",
    ".yaml", ".yml", ".md", ".csv", ".sql", ".sh", ".bat", ".ps1", ".c", ".cpp",
    ".h", ".hpp", ".cs", ".java", ".go", ".rs", ".php", ".rb", ".kt", ".swift",
    ".ini", ".cfg", ".env", ".dockerfile", ".toml", ".log", ".vue", ".svelte"
}


# =====================================================================
# 🗄 БАЗА ДАННЫХ И АВТООЧИСТКА (ТОЛЬКО ТЕКСТОВАЯ ИСТОРИЯ)
# =====================================================================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=FILE;")
    conn.execute("PRAGMA cache_size=-4000;")  # Лимит кэша SQLite ~4 МБ
    return conn


def init_db():
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT DEFAULT 'Новый чат',
                model TEXT DEFAULT ?,
                system_prompt TEXT DEFAULT '',
                created_at REAL
            )
        """, (DEFAULT_MODEL,))
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER,
                role TEXT,
                content TEXT,
                attachments_json TEXT DEFAULT '[]',
                created_at REAL
            )
        """)
        conn.commit()


async def run_web_cleanup():
    now = time.time()
    large_cutoff = now - LARGE_FILE_TTL_SECONDS
    small_cutoff = now - SMALL_FILE_TTL_SECONDS
    chat_cutoff = now - CHAT_TTL_SECONDS
    api_files_to_delete = []

    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, chat_id, attachments_json, created_at FROM messages WHERE attachments_json != '[]'")
        for msg_id, chat_id, att_json, c_ts in c.fetchall():
            atts = json.loads(att_json or "[]")
            changed = False
            for a in atts:
                if a.get("uri"):
                    f_size = a.get("size", 0)
                    ttl_expired = (f_size >= LARGE_FILE_BYTES and c_ts < large_cutoff) or (c_ts < small_cutoff)
                    if ttl_expired:
                        if a.get("api_name"):
                            api_files_to_delete.append(a["api_name"])
                        a["uri"] = None
                        a["api_name"] = None
                        a["expired"] = True
                        changed = True
            if changed:
                c.execute("UPDATE messages SET attachments_json = ? WHERE id = ?", (json.dumps(atts), msg_id))

        # Собираем файлы из чатов старше 7 дней перед их удалением
        c.execute("""
            SELECT attachments_json FROM messages
            WHERE created_at < ? OR chat_id IN (SELECT id FROM chats WHERE created_at < ?)
        """, (chat_cutoff, chat_cutoff))
        for (att_json,) in c.fetchall():
            for a in json.loads(att_json or "[]"):
                if a.get("api_name"):
                    api_files_to_delete.append(a["api_name"])

        # Удаляем чаты и сообщения старше 7 дней
        c.execute("DELETE FROM chats WHERE created_at < ?", (chat_cutoff,))
        c.execute("DELETE FROM messages WHERE created_at < ? OR chat_id NOT IN (SELECT id FROM chats)", (chat_cutoff,))
        conn.commit()

    for f_name in set(api_files_to_delete):
        try:
            await client.aio.files.delete(name=f_name)
        except Exception:
            pass

    with get_db() as conn:
        conn.execute("VACUUM")
    gc.collect()


async def cleanup_loop():
    while True:
        try:
            await run_web_cleanup()
        except Exception as e:
            logger.error(f"Ошибка очистки web БД: {e}")
        await asyncio.sleep(CLEANUP_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    task = asyncio.create_task(cleanup_loop())
    yield
    task.cancel()


app = FastAPI(title="Gemini AI Proxy (1GB RAM Optimized)", lifespan=lifespan)


# =====================================================================
# 🤖 МОДЕЛИ (ОТ НОВЫХ К СТАРЫМ)
# =====================================================================
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


async def get_sorted_models(force: bool = False) -> list[str]:
    global MODELS_CACHE
    if MODELS_CACHE and not force:
        return MODELS_CACHE
    fallback = [
        "gemini-4-carbon", "gemini-4-argon", "gemini-4-hydrogen",
        "gemini-3.8-flash", "gemini-3.5-flash-lite",
        "gemini-2.5-pro", "gemini-2.5-flash", "imagen-3.0-generate-002",
    ]
    try:
        raw = set(fallback)
        pager = await client.aio.models.list()
        async for m in pager:
            name = m.name.replace("models/", "")
            if any(x in name for x in ["embedding", "aqa", "bison", "gecko", "learnlm", "tts"]):
                continue
            if name.startswith("gemini") or name.startswith("imagen"):
                raw.add(name)
        MODELS_CACHE = sorted(list(raw), key=parse_model_sort_key, reverse=True)
        return MODELS_CACHE
    except Exception:
        return sorted(fallback, key=parse_model_sort_key, reverse=True)


# =====================================================================
# 🌐 API ЭНДПОИНТЫ
# =====================================================================
@app.get("/api/models")
async def api_get_models(refresh: bool = False):
    return {"models": await get_sorted_models(force=refresh), "default": DEFAULT_MODEL}


@app.get("/api/chats")
async def api_list_chats():
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, title, model, system_prompt, created_at FROM chats ORDER BY id DESC")
        return {"chats": [{"id": r[0], "title": r[1], "model": r[2], "system_prompt": r[3], "created_at": r[4]} for r in c.fetchall()]}


@app.post("/api/chats")
async def api_create_chat(model: str = Form(DEFAULT_MODEL), system_prompt: str = Form("")):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("INSERT INTO chats (title, model, system_prompt, created_at) VALUES (?, ?, ?, ?)", ("Новый чат", model, system_prompt, time.time()))
        cid = c.lastrowid
        conn.commit()
        return {"id": cid, "title": "Новый чат", "model": model, "system_prompt": system_prompt}


@app.delete("/api/chats/{chat_id}")
async def api_delete_chat(chat_id: int):
    with get_db() as conn:
        conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        conn.commit()
    return {"ok": True}


@app.get("/api/chats/{chat_id}/messages")
async def api_get_chat_messages(chat_id: int):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, title, model, system_prompt FROM chats WHERE id = ?", (chat_id,))
        chat_row = c.fetchone()
        if not chat_row:
            raise HTTPException(status_code=404, detail="Чат не найден или удален по истечении 7 дней")
        c.execute("SELECT id, role, content, attachments_json, created_at FROM messages WHERE chat_id = ? ORDER BY id ASC", (chat_id,))
        msgs = [{"id": r[0], "role": r[1], "content": r[2], "attachments": json.loads(r[3] or "[]"), "created_at": r[4]} for r in c.fetchall()]
        return {"chat": {"id": chat_row[0], "title": chat_row[1], "model": chat_row[2], "system_prompt": chat_row[3]}, "messages": msgs}


@app.post("/api/chat/stream")
async def api_chat_stream(
    chat_id: int = Form(...),
    prompt: str = Form(""),
    model: str = Form(DEFAULT_MODEL),
    system_prompt: str = Form(""),
    temperature: float = Form(1.0),
    thinking_level: str = Form("DEFAULT"),
    use_search: bool = Form(False),
    use_code_exec: bool = Form(False),
    files: List[UploadFile] = File(default=[]),
):
    """
    Потоковая прокачка файлов кусочками по 1 МБ сразу на диск без загрузки в ОЗУ.
    Сразу после отправки файла в Google API локальная копия удаляется.
    """
    saved_temp_files = []
    for uf in files:
        if not uf.filename:
            continue
        ext = Path(uf.filename).suffix.lower()
        fd, t_path = tempfile.mkstemp(suffix=ext or ".bin")
        size = 0
        with os.fdopen(fd, "wb") as out_f:
            while True:
                chunk = await uf.read(CHUNK_SIZE)
                if not chunk:
                    break
                out_f.write(chunk)
                size += len(chunk)
        await uf.close()
        saved_temp_files.append({
            "filename": uf.filename,
            "mime": uf.content_type or "application/octet-stream",
            "ext": ext,
            "path": t_path,
            "size": size,
        })

    async def event_generator():
        attachments_meta = []
        try:
            with get_db() as conn:
                conn.execute("UPDATE chats SET model = ?, system_prompt = ? WHERE id = ?", (model, system_prompt, chat_id))
                conn.commit()

            # 1. Генерация изображений Imagen (показываем в браузере, но в БД пишем только текст)
            if model.startswith("imagen"):
                yield json.dumps({"type": "status", "text": f"🎨 Генерация в {model}..."}) + "\n"
                img_res = await client.aio.models.generate_images(
                    model=model, prompt=prompt or "Art", config=genai_types.GenerateImagesConfig(number_of_images=1)
                )
                for gen_img in img_res.generated_images:
                    b64 = base64.b64encode(gen_img.image.image_bytes).decode("utf-8")
                    yield json.dumps({"type": "image", "mime": "image/png", "data": b64}) + "\n"

                note = f"*(Изображение сгенерировано по запросу: {prompt}. В текстовой истории картинки не хранятся)*"
                yield json.dumps({"type": "chunk", "text": note}) + "\n"
                with get_db() as conn:
                    conn.execute("INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, 'user', ?, ?)", (chat_id, prompt, time.time()))
                    conn.execute("INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, 'model', ?, ?)", (chat_id, note, time.time()))
                    conn.commit()
                yield json.dumps({"type": "done", "title": prompt[:35] or "Генерация фото"}) + "\n"
                return

            # 2. Обработка всех файлов с немедленным удалением с диска
            prompt_with_text = prompt
            for f_item in saved_temp_files:
                fname, mime, ext, t_path, fsize = f_item["filename"], f_item["mime"], f_item["ext"], f_item["path"], f_item["size"]

                if (ext in TEXT_CODE_EXTENSIONS or mime.startswith("text/")) and fsize < 512 * 1024:
                    with open(t_path, "r", encoding="utf-8", errors="replace") as rf:
                        code_str = rf.read()
                    os.remove(t_path)
                    prompt_with_text = f"Файл `{fname}`:\n```{ext.lstrip('.')}\n{code_str}\n```\n\n" + (prompt_with_text or "Проанализируй.")
                    attachments_meta.append({"name": fname, "size": fsize, "uri": None, "api_name": None})
                else:
                    yield json.dumps({"type": "status", "text": f"☁️ Прокачка {fname} ({fsize // 1024} КБ) в Google AI..."}) + "\n"
                    g_file = await client.aio.files.upload(
                        file=t_path, config={"mime_type": mime} if mime != "application/octet-stream" else None
                    )
                    # Мгновенно освобождаем диск VPS
                    if os.path.exists(t_path):
                        os.remove(t_path)

                    while g_file.state.name == "PROCESSING":
                        yield json.dumps({"type": "status", "text": f"🎬 Обработка {fname} на стороне Google..."}) + "\n"
                        await asyncio.sleep(2)
                        g_file = await client.aio.files.get(name=g_file.name)

                    if g_file.state.name == "FAILED":
                        raise RuntimeError(f"Ошибка обработки файла {fname}")

                    attachments_meta.append({
                        "name": fname,
                        "mime": g_file.mime_type or mime,
                        "uri": g_file.uri,
                        "api_name": g_file.name,
                        "size": fsize,
                    })

            if not prompt_with_text and attachments_meta:
                prompt_with_text = "Подробно проанализируй прикрепленные файлы."

            # 3. Запись текстового сообщения в БД
            chat_title = None
            with get_db() as conn:
                c = conn.cursor()
                c.execute(
                    "INSERT INTO messages (chat_id, role, content, attachments_json, created_at) VALUES (?, 'user', ?, ?, ?)",
                    (chat_id, prompt_with_text, json.dumps(attachments_meta), time.time())
                )
                c.execute("SELECT COUNT(*) FROM messages WHERE chat_id = ?", (chat_id,))
                if c.fetchone()[0] <= 1:
                    raw_t = (prompt or (attachments_meta[0]["name"] if attachments_meta else "Диалог")).strip()
                    chat_title = raw_t.replace("\n", " ")[:36]
                    c.execute("UPDATE chats SET title = ? WHERE id = ?", (chat_title, chat_id))
                conn.commit()

            # 4. Сборка истории чата с учетом 24ч для файлов > 10 МБ
            with get_db() as conn:
                c = conn.cursor()
                c.execute("SELECT role, content, attachments_json, created_at FROM messages WHERE chat_id = ? ORDER BY id ASC", (chat_id,))
                rows = c.fetchall()

            contents = []
            now_ts = time.time()
            for r_role, r_txt, r_att_json, r_ts in rows:
                parts = []
                for att in json.loads(r_att_json or "[]"):
                    ttl = LARGE_FILE_TTL_SECONDS if att.get("size", 0) >= LARGE_FILE_BYTES else SMALL_FILE_TTL_SECONDS
                    if att.get("uri") and (now_ts - r_ts) < ttl:
                        parts.append(genai_types.Part.from_uri(file_uri=att["uri"], mime_type=att["mime"]))
                if r_txt:
                    parts.append(genai_types.Part.from_text(text=r_txt))
                if parts:
                    contents.append(genai_types.Content(role=r_role, parts=parts))

            # 5. Настройки генерации
            tools = []
            if use_search:
                tools.append(genai_types.Tool(google_search=genai_types.GoogleSearch()))
            if use_code_exec:
                tools.append(genai_types.Tool(code_execution=genai_types.ToolCodeExecution()))

            cfg = {"temperature": temperature}
            if system_prompt.strip():
                cfg["system_instruction"] = system_prompt.strip()
            if tools:
                cfg["tools"] = tools
            if thinking_level and thinking_level != "DEFAULT":
                cfg["thinking_config"] = genai_types.ThinkingConfig(thinking_level=thinking_level)

            yield json.dumps({"type": "status", "text": f"🧠 {model} думает..."}) + "\n"

            full_text = ""
            stream = await client.aio.models.generate_content_stream(
                model=model, contents=contents, config=genai_types.GenerateContentConfig(**cfg)
            )

            async for chunk in stream:
                if chunk.candidates:
                    for cand in chunk.candidates:
                        if cand.content and cand.content.parts:
                            for part in cand.content.parts:
                                if part.inline_data and part.inline_data.data:
                                    m_type = part.inline_data.mime_type or "image/png"
                                    if m_type.startswith("image/"):
                                        b64 = base64.b64encode(part.inline_data.data).decode("utf-8")
                                        yield json.dumps({"type": "image", "mime": m_type, "data": b64}) + "\n"
                                if getattr(part, "executable_code", None):
                                    cb = f"\n```python\n# [Google Code Execution]\n{part.executable_code.code}\n```\n"
                                    full_text += cb
                                    yield json.dumps({"type": "chunk", "text": cb}) + "\n"
                                if getattr(part, "code_execution_result", None):
                                    ob = f"\n```text\n# [Результат]:\n{part.code_execution_result.output}\n```\n"
                                    full_text += ob
                                    yield json.dumps({"type": "chunk", "text": ob}) + "\n"
                try:
                    if chunk.text:
                        full_text += chunk.text
                        yield json.dumps({"type": "chunk", "text": chunk.text}) + "\n"
                except Exception:
                    pass

            # В БД сохраняется ТОЛЬКО текст ответа
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, 'model', ?, ?)",
                    (chat_id, full_text or "[Медиа-ответ]", time.time())
                )
                conn.commit()

            yield json.dumps({"type": "done", "title": chat_title}) + "\n"

        except Exception as e:
            logger.exception("Ошибка стриминга:")
            yield json.dumps({"type": "error", "text": f"{type(e).__name__}: {str(e)}"}) + "\n"
        finally:
            for item in saved_temp_files:
                if os.path.exists(item["path"]):
                    try:
                        os.remove(item["path"])
                    except OSError:
                        pass
            gc.collect()

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


# =====================================================================
# 🎨 ФРОНТЕНД (ВСТРОЕННЫЙ ИНТЕРФЕЙС CHATGPT / GOOGLE AI STUDIO)
# =====================================================================
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="ru" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Gemini AI Studio & Chat</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <style>
    body { background-color: #131314; color: #e3e3e3; font-family: system-ui, -apple-system, sans-serif; }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-thumb { background: #37393b; border-radius: 3px; }
    .prose pre { background: #1e1f20 !important; padding: 0 !important; border-radius: 0.75rem; overflow: hidden; border: 1px solid #333537; margin: 0.75rem 0; }
    .prose p { margin-bottom: 0.65rem; line-height: 1.6; }
    .prose ul, .prose ol { margin-left: 1.5rem; margin-bottom: 0.65rem; list-style: revert; }
    .gemini-gradient { background: linear-gradient(90deg, #4285f4, #9b72cb, #d96570); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  </style>
</head>
<body class="h-screen flex overflow-hidden">

  <!-- БОКОВАЯ ПАНЕЛЬ ЧАТОВ -->
  <aside id="sidebar" class="w-72 bg-[#1e1f20] flex flex-col border-r border-[#2d2f31] z-20">
    <div class="p-4 flex items-center justify-between">
      <div class="flex items-center gap-2 font-bold text-lg">
        <span>✨</span><span class="gemini-gradient">Gemini AI</span>
      </div>
      <button onclick="createNewChat()" class="px-3 py-1.5 rounded-lg bg-[#282a2c] hover:bg-[#37393b] text-xs font-medium">➕ Новый</button>
    </div>
    <div class="px-3 mb-2">
      <input id="chatSearch" oninput="renderChatList()" type="text" placeholder="🔍 Поиск..."
        class="w-full bg-[#131314] text-xs px-3 py-2 rounded-lg border border-[#333537] focus:outline-none focus:border-blue-500" />
    </div>
    <div id="chatList" class="flex-1 overflow-y-auto px-2 space-y-1"></div>
    <div class="p-3 border-t border-[#2d2f31] text-[11px] text-gray-400 flex justify-between">
      <span>Чаты: 7 дней | >10МБ: 24ч</span>
      <button onclick="exportCurrentChat()" class="hover:text-white underline">Экспорт .md</button>
    </div>
  </aside>

  <!-- ОСНОВНАЯ ОБЛАСТЬ -->
  <main class="flex-1 flex flex-col h-full bg-[#131314]" id="dropZone">
    <header class="h-14 border-b border-[#2d2f31] px-4 flex items-center justify-between">
      <div class="flex items-center gap-2">
        <button onclick="document.getElementById('sidebar').classList.toggle('hidden')" class="p-2 rounded-lg hover:bg-[#282a2c]">☰</button>
        <select id="modelSelect" class="bg-[#1e1f20] text-xs sm:text-sm font-medium px-3 py-1.5 rounded-lg border border-[#37393b] focus:outline-none"></select>
        <button onclick="loadModels(true)" title="Обновить модели" class="p-1.5 rounded-lg hover:bg-[#282a2c] text-sm">🔄</button>
      </div>
      <div class="flex items-center gap-1.5">
        <button onclick="navigateChat(-1)" class="px-2.5 py-1.5 rounded-lg bg-[#1e1f20] hover:bg-[#282a2c] text-xs border border-[#333537]">⬅️</button>
        <button onclick="navigateChat(1)" class="px-2.5 py-1.5 rounded-lg bg-[#1e1f20] hover:bg-[#282a2c] text-xs border border-[#333537]">➡️</button>
        <button onclick="toggleDrawer()" class="px-3 py-1.5 rounded-lg bg-[#1e1f20] hover:bg-[#282a2c] text-xs border border-[#37393b]">⚙️ Настройки</button>
      </div>
    </header>

    <div id="messagesContainer" class="flex-1 overflow-y-auto px-4 py-6 space-y-5 max-w-4xl w-full mx-auto"></div>

    <!-- ПОЛЕ ВВОДА -->
    <div class="p-4 max-w-4xl w-full mx-auto">
      <div id="attachedFilesBar" class="flex flex-wrap gap-2 mb-2 hidden"></div>
      <div class="bg-[#1e1f20] border border-[#37393b] rounded-2xl p-2.5 shadow-lg">
        <textarea id="promptInput" rows="1" placeholder="Введите запрос, вставьте скриншот (Ctrl+V) или перетащите тяжелые файлы..."
          class="w-full bg-transparent resize-none focus:outline-none text-sm px-2 py-1 max-h-44 overflow-y-auto"
          oninput="this.style.height='auto';this.style.height=Math.min(this.scrollHeight,176)+'px'"
          onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendMessage();}"></textarea>
        <div class="flex items-center justify-between mt-2 pt-1.5 border-t border-[#2d2f31]">
          <div class="flex items-center gap-1.5 flex-wrap">
            <input type="file" id="fileInput" multiple class="hidden" onchange="for(const f of this.files)pendingFiles.push(f);this.value='';renderPendingFiles();" />
            <button onclick="document.getElementById('fileInput').click()" class="p-1.5 px-2.5 rounded-xl hover:bg-[#282a2c] text-xs">📎 Файлы</button>
            <button id="micBtn" onclick="toggleVoice()" class="p-1.5 px-2.5 rounded-xl hover:bg-[#282a2c] text-xs">🎤 <span id="micTxt">Голос</span></button>
            <label class="flex items-center gap-1 text-xs text-gray-300 cursor-pointer px-2 py-1 rounded-xl hover:bg-[#282a2c]">
              <input type="checkbox" id="searchToggle" class="accent-blue-500" /> 🔍 Поиск
            </label>
            <label class="flex items-center gap-1 text-xs text-gray-300 cursor-pointer px-2 py-1 rounded-xl hover:bg-[#282a2c]">
              <input type="checkbox" id="codeExecToggle" class="accent-blue-500" /> 🐍 Код
            </label>
          </div>
          <button id="sendBtn" onclick="sendMessage()" class="px-4 py-1.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium disabled:opacity-40">Отправить ➤</button>
        </div>
      </div>
    </div>
  </main>

  <!-- ПРАВАЯ ПАНЕЛЬ НАСТРОЕК AI STUDIO -->
  <aside id="rightDrawer" class="w-72 bg-[#1e1f20] border-l border-[#2d2f31] p-4 flex-col gap-4 hidden">
    <div class="flex justify-between items-center border-b border-[#2d2f31] pb-2">
      <span class="font-semibold text-sm">⚙️ Параметры ИИ</span>
      <button onclick="toggleDrawer()">✕</button>
    </div>
    <div>
      <label class="block text-xs text-gray-400 mb-1">Thinking Level (Размышление)</label>
      <select id="thinkingLevel" class="w-full bg-[#131314] border border-[#37393b] rounded-lg px-2.5 py-1.5 text-xs">
        <option value="DEFAULT">Auto (По умолчанию)</option>
        <option value="LOW">LOW (Быстро)</option>
        <option value="MEDIUM">MEDIUM (Средне)</option>
        <option value="HIGH">HIGH (Глубоко)</option>
        <option value="XHIGH">XHIGH (Максимум Hydrogen)</option>
      </select>
    </div>
    <div>
      <div class="flex justify-between text-xs text-gray-400 mb-1"><span>Температура</span><span id="tempVal">1.0</span></div>
      <input type="range" id="temperatureRange" min="0" max="2" step="0.1" value="1.0" oninput="document.getElementById('tempVal').innerText=this.value" class="w-full accent-blue-500" />
    </div>
    <div>
      <label class="block text-xs text-gray-400 mb-1">System Prompt (Инструкция)</label>
      <textarea id="systemPrompt" rows="5" placeholder="Системная роль..." class="w-full bg-[#131314] border border-[#37393b] rounded-lg p-2 text-xs"></textarea>
    </div>
  </aside>

  <!-- ОКНО LIVE-ПРЕВЬЮ ДЛЯ HTML/JS -->
  <div id="previewModal" class="fixed inset-0 bg-black/80 hidden items-center justify-center z-50 p-6">
    <div class="bg-[#1e1f20] w-full max-w-5xl h-[85vh] rounded-2xl border border-[#37393b] flex flex-col overflow-hidden">
      <div class="px-4 py-2 bg-[#282a2c] flex justify-between items-center">
        <span class="text-xs font-medium">▶️ Live-Превью</span>
        <button onclick="closePreview()" class="px-2 py-1 rounded bg-red-600 text-white text-xs">Закрыть ✕</button>
      </div>
      <iframe id="previewFrame" class="w-full flex-1 bg-white" sandbox="allow-scripts"></iframe>
    </div>
  </div>

<script>
  let chats = [], activeChatId = null, pendingFiles = [], isGenerating = false;
  let mediaRecorder = null, recordedChunks = [], codeStore = {}, codeSeq = 0;
  const extMap = {python:'py',py:'py',javascript:'js',js:'js',typescript:'ts',ts:'ts',html:'html',css:'css',json:'json',sql:'sql',bash:'sh',sh:'sh',c:'c',cpp:'cpp',cs:'cs',java:'java',go:'go',rust:'rs',php:'php',xml:'xml',yaml:'yaml',md:'md'};

  const renderer = new marked.Renderer();
  renderer.code = function(objOrTxt, info) {
    let raw = typeof objOrTxt === 'object' ? objOrTxt.text : objOrTxt;
    let lang = ((typeof objOrTxt === 'object' ? objOrTxt.lang : info) || 'text').toLowerCase().trim();
    const id = 'c_' + (++codeSeq);
    codeStore[id] = { code: raw, lang };
    let hi = raw;
    try { hi = hljs.getLanguage(lang) ? hljs.highlight(raw, {language:lang}).value : hljs.highlightAuto(raw).value; } catch(e){}
    const prevBtn = (lang==='html'||lang==='svg') ? `<button onclick="openPreview('${id}')" class="px-2 py-0.5 rounded bg-emerald-600/30 text-emerald-300">▶️ Превью</button>` : '';
    return `<div class="my-2 rounded-xl overflow-hidden border border-[#37393b] bg-[#1e1f20]">
      <div class="px-3 py-1 bg-[#282a2c] flex justify-between items-center text-xs text-gray-300 border-b border-[#37393b]">
        <span class="font-mono uppercase">${lang}</span>
        <div class="flex gap-2">${prevBtn}
          <button onclick="navigator.clipboard.writeText(codeStore['${id}'].code)">📋 Копировать</button>
          <button onclick="dlText(codeStore['${id}'].code, '${id}.'+(extMap['${lang}']||'txt'))">💾 Скачать</button>
        </div>
      </div>
      <pre class="p-3 overflow-x-auto text-xs font-mono m-0"><code class="hljs">${hi}</code></pre>
    </div>`;
  };
  marked.setOptions({ renderer, breaks: true, gfm: true });

  window.addEventListener('DOMContentLoaded', async () => {
    await loadModels(false);
    await loadChats();
    const dz = document.getElementById('dropZone');
    dz.addEventListener('dragover', e => e.preventDefault());
    dz.addEventListener('drop', e => { e.preventDefault(); for(const f of e.dataTransfer.files) pendingFiles.push(f); renderPendingFiles(); });
    window.addEventListener('paste', e => { if(e.clipboardData?.files.length){ for(const f of e.clipboardData.files) pendingFiles.push(f); renderPendingFiles(); } });
  });

  async function loadModels(refresh) {
    const r = await fetch(`/api/models?refresh=${refresh}`);
    const d = await r.json();
    const s = document.getElementById('modelSelect');
    const cur = s.value || d.default;
    s.innerHTML = d.models.map((m, i) => `<option value="${m}" ${m===cur?'selected':''}>${i<3?'🔥 ':''}${m}</option>`).join('');
  }

  async function loadChats() {
    const r = await fetch('/api/chats');
    chats = (await r.json()).chats;
    if (!chats.length) await createNewChat();
    else { renderChatList(); await selectChat(chats[0].id); }
  }

  function renderChatList() {
    const q = document.getElementById('chatSearch').value.toLowerCase();
    document.getElementById('chatList').innerHTML = chats.filter(c => c.title.toLowerCase().includes(q)).map(c => `
      <div onclick="selectChat(${c.id})" class="group flex justify-between items-center px-3 py-2 rounded-xl cursor-pointer text-xs ${c.id===activeChatId?'bg-[#282a2c] text-white font-medium':'text-gray-400 hover:bg-[#282a2c]/50'}">
        <span class="truncate pr-2">💬 ${esc(c.title)}</span>
        <button onclick="delChat(event,${c.id})" class="opacity-0 group-hover:opacity-100 hover:text-red-400">🗑</button>
      </div>`).join('');
  }

  async function createNewChat() {
    const fd = new FormData();
    fd.append('model', document.getElementById('modelSelect').value || 'gemini-4-hydrogen');
    fd.append('system_prompt', document.getElementById('systemPrompt').value || '');
    const r = await fetch('/api/chats', { method: 'POST', body: fd });
    const c = await r.json();
    chats.unshift(c);
    await selectChat(c.id);
  }

  async function selectChat(id) {
    activeChatId = id;
    renderChatList();
    const r = await fetch(`/api/chats/${id}/messages`);
    const d = await r.json();
    if (d.chat.model) document.getElementById('modelSelect').value = d.chat.model;
    document.getElementById('systemPrompt').value = d.chat.system_prompt || '';
    const box = document.getElementById('messagesContainer');
    box.innerHTML = '';
    if (!d.messages.length) {
      box.innerHTML = `<div id="welcome" class="text-center py-20"><h1 class="text-3xl font-bold mb-2 gemini-gradient">Gemini AI Proxy</h1><p class="text-xs text-gray-400">История хранится только в тексте (7 дней). Файлы > 10 МБ удаляются через 24 часа.</p></div>`;
    } else {
      d.messages.forEach(m => addBubble(m.role, m.content, m.attachments));
      box.scrollTop = box.scrollHeight;
    }
  }

  async function navigateChat(dir) {
    const i = chats.findIndex(c => c.id === activeChatId) - dir;
    if (i >= 0 && i < chats.length) await selectChat(chats[i].id);
  }

  async function delChat(e, id) {
    e.stopPropagation();
    await fetch(`/api/chats/${id}`, { method: 'DELETE' });
    chats = chats.filter(c => c.id !== id);
    if (!chats.length) await createNewChat();
    else if (activeChatId === id) await selectChat(chats[0].id);
    else renderChatList();
  }

  function renderPendingFiles() {
    const b = document.getElementById('attachedFilesBar');
    if (!pendingFiles.length) { b.classList.add('hidden'); return; }
    b.classList.remove('hidden');
    b.innerHTML = pendingFiles.map((f, i) => `<div class="flex items-center gap-1.5 bg-[#282a2c] border border-[#37393b] px-2.5 py-1 rounded-lg text-xs">
      <span>📎 ${esc(f.name)} (${(f.size/1048576).toFixed(1)} МБ)</span>
      <button onclick="pendingFiles.splice(${i},1);renderPendingFiles()" class="hover:text-red-400">✕</button>
    </div>`).join('');
  }

  async function toggleVoice() {
    const st = document.getElementById('micTxt');
    if (mediaRecorder && mediaRecorder.state === 'recording') { mediaRecorder.stop(); st.textContent = 'Голос'; return; }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
    mediaRecorder.onstop = () => {
      pendingFiles.push(new File([new Blob(recordedChunks, {type:'audio/webm'})], `voice_${Date.now()}.webm`, {type:'audio/webm'}));
      renderPendingFiles();
      stream.getTracks().forEach(t => t.stop());
    };
    mediaRecorder.start();
    st.textContent = 'Стоп ●';
  }

  async function sendMessage() {
    if (isGenerating) return;
    const inp = document.getElementById('promptInput');
    const txt = inp.value.trim();
    if (!txt && !pendingFiles.length) return;
    if (!activeChatId) await createNewChat();

    isGenerating = true;
    document.getElementById('sendBtn').disabled = true;
    document.getElementById('welcome')?.remove();

    const files = [...pendingFiles];
    inp.value = ''; inp.style.height = 'auto';
    pendingFiles = []; renderPendingFiles();

    addBubble('user', txt, files.map(f => ({name: f.name})));
    const ai = createAiBubble();
    const box = document.getElementById('messagesContainer');
    box.scrollTop = box.scrollHeight;

    const fd = new FormData();
    fd.append('chat_id', activeChatId);
    fd.append('prompt', txt);
    fd.append('model', document.getElementById('modelSelect').value);
    fd.append('system_prompt', document.getElementById('systemPrompt').value);
    fd.append('temperature', document.getElementById('temperatureRange').value);
    fd.append('thinking_level', document.getElementById('thinkingLevel').value);
    fd.append('use_search', document.getElementById('searchToggle').checked);
    fd.append('use_code_exec', document.getElementById('codeExecToggle').checked);
    files.forEach(f => fd.append('files', f));

    let full = '';
    try {
      const res = await fetch('/api/chat/stream', { method: 'POST', body: fd });
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop();
        for (const l of lines) {
          if (!l.trim()) continue;
          const ev = JSON.parse(l);
          if (ev.type === 'status') { ai.st.textContent = ev.text; ai.st.classList.remove('hidden'); }
          else if (ev.type === 'chunk') { ai.st.classList.add('hidden'); full += ev.text; ai.cnt.innerHTML = marked.parse(full); box.scrollTop = box.scrollHeight; }
          else if (ev.type === 'image') {
            ai.st.classList.add('hidden');
            ai.img.innerHTML += `<img src="data:${ev.mime};base64,${ev.data}" class="max-w-sm rounded-xl border border-[#37393b]" />`;
          }
          else if (ev.type === 'done') {
            ai.st.classList.add('hidden'); ai.raw = full; ai.act.classList.remove('hidden');
            if (ev.title) { const c = chats.find(x => x.id === activeChatId); if (c) { c.title = ev.title; renderChatList(); } }
          }
          else if (ev.type === 'error') { ai.st.classList.add('hidden'); ai.cnt.innerHTML = `<span class="text-red-400 text-xs">❌ ${esc(ev.text)}</span>`; }
        }
      }
    } catch (e) {
      ai.cnt.innerHTML = `<span class="text-red-400 text-xs">❌ ${esc(e.message)}</span>`;
    } finally {
      isGenerating = false;
      document.getElementById('sendBtn').disabled = false;
    }
  }

  function addBubble(role, txt, atts = []) {
    const box = document.getElementById('messagesContainer');
    if (role === 'user') {
      const aHtml = (atts||[]).map(a => `<span class="bg-[#131314] border border-[#37393b] px-2 py-0.5 rounded text-xs">📎 ${esc(a.name)}</span>`).join(' ');
      const d = document.createElement('div');
      d.className = 'flex justify-end';
      d.innerHTML = `<div class="max-w-2xl bg-[#282a2c] px-4 py-2.5 rounded-2xl rounded-tr-sm text-sm space-y-1.5">${aHtml?`<div>${aHtml}</div>`:''}<div class="whitespace-pre-wrap">${esc(txt)}</div></div>`;
      box.appendChild(d);
    } else {
      const b = createAiBubble();
      b.st.classList.add('hidden'); b.raw = txt || '';
      b.cnt.innerHTML = marked.parse(txt || '');
      b.act.classList.remove('hidden');
    }
  }

  function createAiBubble() {
    const box = document.getElementById('messagesContainer');
    const d = document.createElement('div');
    d.className = 'flex gap-3 items-start';
    const o = { raw: '' };
    d.innerHTML = `<div class="w-7 h-7 rounded-full bg-blue-600 flex items-center justify-center text-xs shrink-0 mt-1">✨</div>
      <div class="flex-1 min-w-0 space-y-2">
        <div class="st text-xs text-blue-400 animate-pulse">⏳ Обработка...</div>
        <div class="img flex flex-wrap gap-2"></div>
        <div class="cnt prose prose-invert max-w-none text-sm"></div>
        <div class="act hidden flex gap-2 text-xs text-gray-400">
          <button class="hover:text-white bc">📋 Копировать</button>
          <button class="hover:text-white bm">📝 .md</button>
          <button class="hover:text-white bt">📄 .txt</button>
        </div>
      </div>`;
    box.appendChild(d);
    o.st = d.querySelector('.st'); o.img = d.querySelector('.img'); o.cnt = d.querySelector('.cnt'); o.act = d.querySelector('.act');
    d.querySelector('.bc').onclick = () => navigator.clipboard.writeText(o.raw);
    d.querySelector('.bm').onclick = () => dlText(o.raw, `ans_${Date.now()}.md`);
    d.querySelector('.bt').onclick = () => dlText(o.raw, `ans_${Date.now()}.txt`);
    return o;
  }

  function openPreview(id) {
    document.getElementById('previewFrame').srcdoc = codeStore[id].code;
    const m = document.getElementById('previewModal'); m.classList.remove('hidden'); m.classList.add('flex');
  }
  function closePreview() {
    const m = document.getElementById('previewModal'); m.classList.add('hidden'); m.classList.remove('flex');
  }
  function dlText(t, n) {
    const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([t], {type:'text/plain;charset=utf-8'})); a.download = n; a.click();
  }
  async function exportCurrentChat() {
    if (!activeChatId) return;
    const r = await fetch(`/api/chats/${activeChatId}/messages`);
    const d = await r.json();
    dlText(d.messages.map(m => `### ${m.role}\n\n${m.content}`).join('\n\n---\n\n'), `chat_${activeChatId}.md`);
  }
  function toggleDrawer() { const d = document.getElementById('rightDrawer'); d.classList.toggle('hidden'); d.classList.toggle('flex'); }
  function esc(s) { return (s||'').replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m])); }
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return HTML_PAGE


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        workers=1,
        limit_concurrency=20,
        timeout_keep_alive=5,
    )
