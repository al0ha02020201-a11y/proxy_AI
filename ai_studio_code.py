import asyncio
import base64
import json
import logging
import os
import re
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from google import genai
from google.genai import types as genai_types

# =====================================================================
# ⚙️ НАСТРОЙКИ
# =====================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "ВАШ_КЛЮЧ_GEMINI_API")
DEFAULT_MODEL = "gemini-4-hydrogen"
DB_PATH = "gemini_web.db"
HOST = "0.0.0.0"
PORT = 8000

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Gemini AI Web Proxy")
client = genai.Client(api_key=GEMINI_API_KEY)

MODELS_CACHE = []

TEXT_CODE_EXTENSIONS = {
    ".txt", ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".json", ".xml",
    ".yaml", ".yml", ".md", ".csv", ".sql", ".sh", ".bat", ".ps1", ".c", ".cpp",
    ".h", ".hpp", ".cs", ".java", ".go", ".rs", ".php", ".rb", ".kt", ".swift",
    ".ini", ".cfg", ".env", ".dockerfile", ".toml", ".log", ".vue", ".svelte"
}


# =====================================================================
# 🗄 БАЗА ДАННЫХ (SQLite)
# =====================================================================
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT DEFAULT 'Новый чат',
                model TEXT DEFAULT ?,
                system_prompt TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """, (DEFAULT_MODEL,))
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER,
                role TEXT,
                content TEXT,
                attachments_json TEXT DEFAULT '[]',
                images_json TEXT DEFAULT '[]',
                created_at REAL
            )
        """)
        conn.commit()


# =====================================================================
# 🤖 СОРТИРОВКА МОДЕЛЕЙ (ОТ САМЫХ НОВЫХ К СТАРЫМ)
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
    if "carbon" in model_name:
        tier = 6
    elif "argon" in model_name:
        tier = 5
    elif "hydrogen" in model_name:
        tier = 4
    elif "pro" in model_name:
        tier = 3
    elif "flash" in model_name and "lite" not in model_name:
        tier = 2

    is_latest = 1 if ("latest" in model_name or "preview" in model_name or "exp" in model_name) else 0
    return (major, minor, tier, is_latest, model_name)


async def get_sorted_models(force: bool = False) -> list[str]:
    global MODELS_CACHE
    if MODELS_CACHE and not force:
        return MODELS_CACHE

    fallback = [
        "gemini-4-carbon",
        "gemini-4-argon",
        "gemini-4-hydrogen",
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "imagen-3.0-generate-002",
    ]
    try:
        raw_models = set(fallback)
        pager = await client.aio.models.list()
        async for m in pager:
            name = m.name.replace("models/", "")
            if any(x in name for x in ["embedding", "aqa", "bison", "gecko", "learnlm", "tts"]):
                continue
            if name.startswith("gemini") or name.startswith("imagen"):
                raw_models.append(name) if isinstance(raw_models, list) else raw_models.add(name)

        MODELS_CACHE = sorted(list(raw_models), key=parse_model_sort_key, reverse=True)
        return MODELS_CACHE
    except Exception as e:
        logger.warning(f"Не удалось получить список моделей из API, используем базовый: {e}")
        return sorted(fallback, key=parse_model_sort_key, reverse=True)


# =====================================================================
# 🌐 API ЭНДПОИНТЫ
# =====================================================================
@app.get("/api/models")
async def api_get_models(refresh: bool = False):
    models = await get_sorted_models(force=refresh)
    return {"models": models, "default": DEFAULT_MODEL}


@app.get("/api/chats")
async def api_list_chats():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT id, title, model, system_prompt, created_at FROM chats ORDER BY id DESC")
        rows = c.fetchall()
        return {
            "chats": [
                {"id": r[0], "title": r[1], "model": r[2], "system_prompt": r[3], "created_at": r[4]}
                for r in rows
            ]
        }


@app.post("/api/chats")
async def api_create_chat(model: str = Form(DEFAULT_MODEL), system_prompt: str = Form("")):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO chats (title, model, system_prompt) VALUES (?, ?, ?)",
            ("Новый чат", model, system_prompt)
        )
        chat_id = c.lastrowid
        conn.commit()
        return {"id": chat_id, "title": "Новый чат", "model": model, "system_prompt": system_prompt}


@app.delete("/api/chats/{chat_id}")
async def api_delete_chat(chat_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        c.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        conn.commit()
    return {"ok": True}


@app.get("/api/chats/{chat_id}/messages")
async def api_get_chat_messages(chat_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT id, title, model, system_prompt FROM chats WHERE id = ?", (chat_id,))
        chat_row = c.fetchone()
        if not chat_row:
            raise HTTPException(status_code=404, detail="Чат не найден")

        c.execute(
            "SELECT id, role, content, attachments_json, images_json, created_at FROM messages WHERE chat_id = ? ORDER BY id ASC",
            (chat_id,)
        )
        msgs = [
            {
                "id": r[0],
                "role": r[1],
                "content": r[2],
                "attachments": json.loads(r[3] or "[]"),
                "images": json.loads(r[4] or "[]"),
                "created_at": r[5],
            }
            for r in c.fetchall()
        ]
        return {
            "chat": {"id": chat_row[0], "title": chat_row[1], "model": chat_row[2], "system_prompt": chat_row[3]},
            "messages": msgs,
        }


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
    """Потоковый эндпоинт (NDJSON): принимает файлы + текст и стримит ответ в реальном времени."""

    # Предварительно считываем переданные файлы в память до старта генератора
    uploaded_files_raw = []
    for uf in files:
        if uf.filename:
            data = await uf.read()
            uploaded_files_raw.append({
                "filename": uf.filename,
                "content_type": uf.content_type or "application/octet-stream",
                "bytes": data,
            })

    async def event_generator():
        temp_paths = []
        attachments_meta = []

        try:
            # Обновляем выбранную модель и system_prompt у чата
            with sqlite3.connect(DB_PATH) as conn:
                c = conn.cursor()
                c.execute("UPDATE chats SET model = ?, system_prompt = ? WHERE id = ?", (model, system_prompt, chat_id))
                conn.commit()

            # 1. Если выбрана модель генерации картинок Imagen
            if model.startswith("imagen"):
                yield json.dumps({"type": "status", "text": f"🎨 Генерация изображения в {model}..."}) + "\n"
                with sqlite3.connect(DB_PATH) as conn:
                    c = conn.cursor()
                    c.execute(
                        "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, 'user', ?, ?)",
                        (chat_id, prompt, time.time())
                    )
                    conn.commit()

                img_res = await client.aio.models.generate_images(
                    model=model,
                    prompt=prompt or "Futuristic digital art",
                    config=genai_types.GenerateImagesConfig(number_of_images=1)
                )
                b64_images = []
                for gen_img in img_res.generated_images:
                    b64 = base64.b64encode(gen_img.image.image_bytes).decode("utf-8")
                    b64_images.append({"mime": "image/png", "data": b64})
                    yield json.dumps({"type": "image", "mime": "image/png", "data": b64}) + "\n"

                caption = f"Сгенерировано изображение по запросу: *{prompt}*"
                yield json.dumps({"type": "chunk", "text": caption}) + "\n"

                with sqlite3.connect(DB_PATH) as conn:
                    c = conn.cursor()
                    c.execute(
                        "INSERT INTO messages (chat_id, role, content, images_json, created_at) VALUES (?, 'model', ?, ?, ?)",
                        (chat_id, caption, json.dumps(b64_images), time.time())
                    )
                    conn.commit()

                yield json.dumps({"type": "done", "title": prompt[:35] or "Генерация фото"}) + "\n"
                return

            # 2. Обработка всех прикрепленных файлов (текст, код, видео, аудио, PDF, фото)
            prompt_with_text_files = prompt
            for f_item in uploaded_files_raw:
                fname = f_item["filename"]
                mime = f_item["content_type"]
                raw_bytes = f_item["bytes"]
                ext = Path(fname).suffix.lower()

                # Если это код или текстовый файл — встраиваем как текст мгновенно
                if ext in TEXT_CODE_EXTENSIONS or mime.startswith("text/"):
                    text_str = raw_bytes.decode("utf-8", errors="replace")
                    prompt_with_text_files = (
                        f"Файл `{fname}`:\n```{ext.lstrip('.')}\n{text_str}\n```\n\n"
                        + (prompt_with_text_files or "Проанализируй этот файл.")
                    )
                    attachments_meta.append({"name": fname, "mime": mime, "uri": None, "type": "code"})
                else:
                    # Иначе загружаем через Gemini File API
                    yield json.dumps({"type": "status", "text": f"☁️ Загрузка {fname} в Google AI..."}) + "\n"
                    fd, t_path = tempfile.mkstemp(suffix=ext or ".bin")
                    os.close(fd)
                    with open(t_path, "wb") as out_f:
                        out_f.write(raw_bytes)
                    temp_paths.append(t_path)

                    g_file = await client.aio.files.upload(
                        file=t_path,
                        config={"mime_type": mime} if mime != "application/octet-stream" else None
                    )
                    while g_file.state.name == "PROCESSING":
                        yield json.dumps({"type": "status", "text": f"🎬 Обработка медиафайла {fname}..."}) + "\n"
                        await asyncio.sleep(2.5)
                        g_file = await client.aio.files.get(name=g_file.name)

                    if g_file.state.name == "FAILED":
                        raise RuntimeError(f"Google AI не смог обработать файл {fname}")

                    attachments_meta.append({
                        "name": fname,
                        "mime": g_file.mime_type or mime,
                        "uri": g_file.uri,
                        "type": "media"
                    })

            if not prompt_with_text_files and attachments_meta:
                prompt_with_text_files = "Подробно проанализируй прикрепленные файлы."

            # 3. Сохраняем сообщение пользователя в БД и обновляем заголовок чата, если оно первое
            chat_title = None
            with sqlite3.connect(DB_PATH) as conn:
                c = conn.cursor()
                c.execute(
                    "INSERT INTO messages (chat_id, role, content, attachments_json, created_at) VALUES (?, 'user', ?, ?, ?)",
                    (chat_id, prompt_with_text_files, json.dumps(attachments_meta), time.time())
                )
                c.execute("SELECT COUNT(*) FROM messages WHERE chat_id = ?", (chat_id,))
                if c.fetchone()[0] <= 1:
                    raw_title = (prompt or (attachments_meta[0]["name"] if attachments_meta else "Новый диалог")).strip()
                    chat_title = raw_title.replace("\n", " ")[:38]
                    c.execute("UPDATE chats SET title = ? WHERE id = ?", (chat_title, chat_id))
                conn.commit()

            # 4. Собираем историю сообщений из БД для контекста
            with sqlite3.connect(DB_PATH) as conn:
                c = conn.cursor()
                c.execute(
                    "SELECT role, content, attachments_json, created_at FROM messages WHERE chat_id = ? ORDER BY id ASC",
                    (chat_id,)
                )
                history_rows = c.fetchall()

            contents = []
            now_ts = time.time()
            for r_role, r_text, r_att_json, r_ts in history_rows:
                parts = []
                att_list = json.loads(r_att_json or "[]")
                for att in att_list:
                    # Ссылки на файлы в Gemini File API активны 48 часов
                    if att.get("uri") and (now_ts - r_ts) < 170000:
                        parts.append(genai_types.Part.from_uri(file_uri=att["uri"], mime_type=att["mime"]))
                if r_text:
                    parts.append(genai_types.Part.from_text(text=r_text))
                if parts:
                    contents.append(genai_types.Content(role=r_role, parts=parts))

            # 5. Формируем конфигурацию генерации (Thinking, Search, Code Execution, System Prompt)
            tools = []
            if use_search:
                tools.append(genai_types.Tool(google_search=genai_types.GoogleSearch()))
            if use_code_exec:
                tools.append(genai_types.Tool(code_execution=genai_types.ToolCodeExecution()))

            config_kwargs = {"temperature": temperature}
            if system_prompt.strip():
                config_kwargs["system_instruction"] = system_prompt.strip()
            if tools:
                config_kwargs["tools"] = tools
            if thinking_level and thinking_level != "DEFAULT":
                config_kwargs["thinking_config"] = genai_types.ThinkingConfig(thinking_level=thinking_level)

            gen_config = genai_types.GenerateContentConfig(**config_kwargs)

            yield json.dumps({"type": "status", "text": f"🧠 {model} думает..."}) + "\n"

            # 6. Запускаем потоковую генерацию
            full_response_text = ""
            generated_images = []

            stream = await client.aio.models.generate_content_stream(
                model=model,
                contents=contents,
                config=gen_config
            )

            async for chunk in stream:
                # Проверяем наличие сгенерированных картинок или кода внутри частей ответа
                if chunk.candidates:
                    for cand in chunk.candidates:
                        if cand.content and cand.content.parts:
                            for part in cand.content.parts:
                                if part.inline_data and part.inline_data.data:
                                    m_type = part.inline_data.mime_type or "image/png"
                                    b64_str = base64.b64encode(part.inline_data.data).decode("utf-8")
                                    if m_type.startswith("image/"):
                                        generated_images.append({"mime": m_type, "data": b64_str})
                                        yield json.dumps({"type": "image", "mime": m_type, "data": b64_str}) + "\n"
                                if getattr(part, "executable_code", None):
                                    code_block = f"\n```python\n# [Выполнение кода Google AI]\n{part.executable_code.code}\n```\n"
                                    full_response_text += code_block
                                    yield json.dumps({"type": "chunk", "text": code_block}) + "\n"
                                if getattr(part, "code_execution_result", None):
                                    out_block = f"\n```text\n# [Результат выполнения]:\n{part.code_execution_result.output}\n```\n"
                                    full_response_text += out_block
                                    yield json.dumps({"type": "chunk", "text": out_block}) + "\n"

                try:
                    if chunk.text:
                        full_response_text += chunk.text
                        yield json.dumps({"type": "chunk", "text": chunk.text}) + "\n"
                except Exception:
                    pass

            # 7. Сохраняем ответ ИИ в БД
            with sqlite3.connect(DB_PATH) as conn:
                c = conn.cursor()
                c.execute(
                    "INSERT INTO messages (chat_id, role, content, images_json, created_at) VALUES (?, 'model', ?, ?, ?)",
                    (chat_id, full_response_text or "[Медиа-ответ]", json.dumps(generated_images), time.time())
                )
                conn.commit()

            yield json.dumps({"type": "done", "title": chat_title}) + "\n"

        except Exception as e:
            logger.exception("Ошибка генерации:")
            yield json.dumps({"type": "error", "text": f"{type(e).__name__}: {str(e)}"}) + "\n"
        finally:
            for p in temp_paths:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


# =====================================================================
# 🎨 ФРОНТЕНД (ВСТРОЕННЫЙ ИНТЕРФЕЙС В СТИЛЕ CHATGPT / GOOGLE AI)
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
    body { background-color: #131314; color: #e3e3e3; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #37393b; border-radius: 3px; }
    .prose pre { background: #1e1f20 !important; padding: 0 !important; border-radius: 0.75rem; overflow: hidden; border: 1px solid #333537; margin: 1rem 0; }
    .prose code { color: #8ab4f8; }
    .prose p { margin-bottom: 0.75rem; line-height: 1.65; }
    .prose ul, .prose ol { margin-left: 1.5rem; margin-bottom: 0.75rem; list-style: revert; }
    .prose table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
    .prose th, .prose td { border: 1px solid #37393b; padding: 0.5rem 0.75rem; }
    .prose th { background: #1e1f20; }
    .gemini-gradient { background: linear-gradient(90deg, #4285f4, #9b72cb, #d96570); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  </style>
</head>
<body class="h-screen flex overflow-hidden select-text">

  <!-- ЛЕВАЯ ПАНЕЛЬ: ИСТОРИЯ ЧАТОВ (КАК В CHATGPT / GEMINI) -->
  <aside id="sidebar" class="w-72 bg-[#1e1f20] flex flex-col border-r border-[#2d2f31] transition-all duration-200 z-20">
    <div class="p-4 flex items-center justify-between">
      <div class="flex items-center gap-2 font-bold text-lg">
        <span class="text-2xl">✨</span>
        <span class="gemini-gradient">Gemini AI</span>
      </div>
      <button onclick="createNewChat()" title="Новый чат" class="px-3 py-1.5 rounded-lg bg-[#282a2c] hover:bg-[#37393b] text-sm font-medium flex items-center gap-1.5 transition">
        <span>➕</span> Новый
      </button>
    </div>

    <div class="px-3 mb-2">
      <input id="chatSearch" oninput="renderChatList()" type="text" placeholder="🔍 Поиск по чатам..."
        class="w-full bg-[#131314] text-sm px-3 py-2 rounded-lg border border-[#333537] focus:outline-none focus:border-blue-500" />
    </div>

    <div id="chatList" class="flex-1 overflow-y-auto px-2 space-y-1"></div>

    <div class="p-3 border-t border-[#2d2f31] text-xs text-gray-400 flex justify-between items-center">
      <span>Мультимодальный прокси</span>
      <button onclick="exportCurrentChat()" class="hover:text-white underline">Экспорт чата</button>
    </div>
  </aside>

  <!-- ЦЕНТРАЛЬНАЯ ОБЛАСТЬ -->
  <main class="flex-1 flex flex-col h-full relative bg-[#131314]" id="dropZone">

    <!-- ВЕРХНЯЯ ПАНЕЛЬ С ВЫБОРОМ МОДЕЛИ И КНОПКОЙ НАСТРОЕК AI STUDIO -->
    <header class="h-14 border-b border-[#2d2f31] px-4 flex items-center justify-between bg-[#131314]/90 backdrop-blur">
      <div class="flex items-center gap-3">
        <button onclick="toggleSidebar()" class="p-2 rounded-lg hover:bg-[#282a2c] text-gray-300">☰</button>
        <div class="flex items-center gap-2">
          <label class="text-xs text-gray-400 hidden sm:inline">Модель:</label>
          <select id="modelSelect" onchange="onModelChange()"
            class="bg-[#1e1f20] text-sm font-medium px-3 py-1.5 rounded-lg border border-[#37393b] focus:outline-none focus:border-blue-500 cursor-pointer">
          </select>
          <button onclick="loadModels(true)" title="Обновить список моделей из API" class="p-1.5 rounded-lg hover:bg-[#282a2c] text-gray-400 hover:text-white text-sm">🔄</button>
        </div>
      </div>

      <div class="flex items-center gap-2">
        <button onclick="navigateAdjacentChat(-1)" title="Прошлый чат" class="px-2.5 py-1.5 rounded-lg bg-[#1e1f20] hover:bg-[#282a2c] text-xs border border-[#333537]">⬅️ Прошлый</button>
        <button onclick="navigateAdjacentChat(1)" title="Следующий чат" class="px-2.5 py-1.5 rounded-lg bg-[#1e1f20] hover:bg-[#282a2c] text-xs border border-[#333537]">Следующий ➡️</button>
        <button onclick="toggleRightDrawer()" class="px-3 py-1.5 rounded-lg bg-[#1e1f20] hover:bg-[#282a2c] text-xs font-medium border border-[#37393b] flex items-center gap-1.5">
          ⚙️ <span class="hidden md:inline">Параметры AI Studio</span>
        </button>
      </div>
    </header>

    <!-- ЛЕНТА СООБЩЕНИЙ -->
    <div id="messagesContainer" class="flex-1 overflow-y-auto px-4 py-6 space-y-6 max-w-4xl w-full mx-auto">
      <div id="welcomeScreen" class="h-full flex flex-col items-center justify-center text-center py-16">
        <h1 class="text-4xl font-bold mb-3 gemini-gradient">Чем я могу помочь?</h1>
        <p class="text-gray-400 max-w-lg text-sm mb-8">
          Отправляйте текст, код, фото, видео, голосовые или PDF. Перетаскивайте файлы прямо сюда или вставляйте скриншоты через <b>Ctrl+V</b>.
        </p>
        <div class="grid grid-cols-1 sm:grid-cols-3 gap-3 max-w-2xl w-full text-left">
          <button onclick="quickPrompt('Напиши современную змейку на HTML5 Canvas в одном файле')" class="p-3.5 rounded-xl bg-[#1e1f20] hover:bg-[#282a2c] border border-[#333537] transition">
            <div class="text-lg mb-1">🎮 Код + Live Превью</div>
            <div class="text-xs text-gray-400">Создать игру на HTML5 и запустить прямо в браузере</div>
          </button>
          <button onclick="quickPrompt('Сравни архитектуры трансформеров и мамба-моделей в таблице')" class="p-3.5 rounded-xl bg-[#1e1f20] hover:bg-[#282a2c] border border-[#333537] transition">
            <div class="text-lg mb-1">🧠 Глубокий разбор</div>
            <div class="text-xs text-gray-400">Аналитика с режимом мышления Gemini 4 Hydrogen</div>
          </button>
          <button onclick="document.getElementById('fileInput').click()" class="p-3.5 rounded-xl bg-[#1e1f20] hover:bg-[#282a2c] border border-[#333537] transition">
            <div class="text-lg mb-1">📎 Анализ любых файлов</div>
            <div class="text-xs text-gray-400">Загрузить видео, аудио, PDF или архив с кодом</div>
          </button>
        </div>
      </div>
    </div>

    <!-- НИЖНЯЯ ПАНЕЛЬ ВВОДА (КОМПОЗЕР) -->
    <div class="p-4 max-w-4xl w-full mx-auto">
      <!-- Список прикрепленных файлов -->
      <div id="attachedFilesBar" class="flex flex-wrap gap-2 mb-2 hidden"></div>

      <div class="bg-[#1e1f20] border border-[#37393b] rounded-2xl p-2.5 shadow-lg focus-within:border-blue-500 transition">
        <textarea id="promptInput" rows="1" placeholder="Спросите Gemini или перетащите файлы сюда..."
          class="w-full bg-transparent resize-none focus:outline-none text-sm px-2 py-1 max-h-48 overflow-y-auto"
          oninput="autoResizeTextarea(this)" onkeydown="handleEnterKey(event)"></textarea>

        <div class="flex items-center justify-between mt-2 pt-1.5 border-t border-[#2d2f31]">
          <div class="flex items-center gap-1.5">
            <input type="file" id="fileInput" multiple class="hidden" onchange="handleFileSelect(event)" />
            <button onclick="document.getElementById('fileInput').click()" title="Прикрепить любые файлы (код, видео, аудио, фото, PDF)"
              class="p-2 rounded-xl hover:bg-[#282a2c] text-gray-300 flex items-center gap-1 text-xs">
              📎 <span class="hidden sm:inline">Файлы</span>
            </button>

            <button id="micBtn" onclick="toggleVoiceRecording()" title="Записать голосовое сообщение с микрофона"
              class="p-2 rounded-xl hover:bg-[#282a2c] text-gray-300 flex items-center gap-1 text-xs">
              🎤 <span id="micStatus" class="hidden sm:inline">Голос</span>
            </button>

            <label class="flex items-center gap-1.5 text-xs text-gray-300 cursor-pointer px-2 py-1.5 rounded-xl hover:bg-[#282a2c]" title="Искать актуальную информацию в Google">
              <input type="checkbox" id="searchToggle" class="accent-blue-500" />
              🔍 <span class="hidden sm:inline">Google Поиск</span>
            </label>

            <label class="flex items-center gap-1.5 text-xs text-gray-300 cursor-pointer px-2 py-1.5 rounded-xl hover:bg-[#282a2c]" title="Выполнять Python-код на серверах Google">
              <input type="checkbox" id="codeExecToggle" class="accent-blue-500" />
              🐍 <span class="hidden sm:inline">Запуск кода</span>
            </label>
          </div>

          <button id="sendBtn" onclick="sendMessage()"
            class="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-medium text-sm flex items-center gap-1.5 transition disabled:opacity-40">
            <span>Отправить</span> ➤
          </button>
        </div>
      </div>
    </div>
  </main>

  <!-- ПРАВАЯ ПАНЕЛЬ НАСТРОЕК (В СТИЛЕ GOOGLE AI STUDIO) -->
  <aside id="rightDrawer" class="w-80 bg-[#1e1f20] border-l border-[#2d2f31] p-4 flex-col gap-5 hidden overflow-y-auto">
    <div class="flex items-center justify-between border-b border-[#2d2f31] pb-3">
      <h2 class="font-semibold text-sm">⚙️ Настройки генерации</h2>
      <button onclick="toggleRightDrawer()" class="text-gray-400 hover:text-white">✕</button>
    </div>

    <div>
      <label class="block text-xs text-gray-400 mb-1.5">Уровень размышлений (Thinking Level)</label>
      <select id="thinkingLevel" class="w-full bg-[#131314] border border-[#37393b] rounded-lg px-3 py-2 text-sm">
        <option value="DEFAULT">По умолчанию (Auto / HIGH)</option>
        <option value="LOW">LOW (Максимальная скорость)</option>
        <option value="MEDIUM">MEDIUM (Баланс)</option>
        <option value="HIGH">HIGH (Глубокое рассуждение)</option>
        <option value="XHIGH">XHIGH (Максимум для Hydrogen)</option>
      </select>
    </div>

    <div>
      <div class="flex justify-between text-xs text-gray-400 mb-1">
        <span>Температура (Креативность)</span>
        <span id="tempVal">1.0</span>
      </div>
      <input type="range" id="temperatureRange" min="0" max="2" step="0.1" value="1.0"
        oninput="document.getElementById('tempVal').innerText = this.value"
        class="w-full accent-blue-500" />
    </div>

    <div class="flex-1 flex flex-col">
      <label class="block text-xs text-gray-400 mb-1.5">System Instructions (Системный промпт)</label>
      <textarea id="systemPrompt" rows="6" placeholder="Например: Ты сеньор Python-разработчик. Пиши чистый, типизированный код без лишней воды..."
        class="w-full bg-[#131314] border border-[#37393b] rounded-lg p-2.5 text-xs focus:outline-none focus:border-blue-500 resize-y"></textarea>
    </div>
  </aside>

  <!-- МОДАЛЬНОЕ ОКНО ДЛЯ LIVE-ПРЕВЬЮ HTML/JS КОДА -->
  <div id="previewModal" class="fixed inset-0 bg-black/80 hidden items-center justify-center z-50 p-6">
    <div class="bg-[#1e1f20] w-full max-w-5xl h-[85vh] rounded-2xl border border-[#37393b] flex flex-col overflow-hidden">
      <div class="px-4 py-2.5 bg-[#282a2c] flex items-center justify-between border-b border-[#37393b]">
        <span class="text-sm font-medium">▶️ Интерактивное превью кода (Sandbox)</span>
        <button onclick="closePreviewModal()" class="px-2.5 py-1 rounded bg-red-600/80 hover:bg-red-600 text-white text-xs">Закрыть ✕</button>
      </div>
      <iframe id="previewFrame" class="w-full flex-1 bg-white" sandbox="allow-scripts allow-modals"></iframe>
    </div>
  </div>

<script>
  let chats = [];
  let activeChatId = null;
  let pendingFiles = [];
  let isGenerating = false;
  let mediaRecorder = null;
  let recordedChunks = [];
  let codeBlocksStore = {};
  let codeBlockCounter = 0;

  const langExtMap = {
    python: 'py', py: 'py', javascript: 'js', js: 'js', typescript: 'ts', ts: 'ts',
    html: 'html', css: 'css', json: 'json', sql: 'sql', bash: 'sh', sh: 'sh',
    c: 'c', cpp: 'cpp', csharp: 'cs', cs: 'cs', java: 'java', go: 'go',
    rust: 'rs', php: 'php', ruby: 'rb', kotlin: 'kt', swift: 'swift',
    xml: 'xml', yaml: 'yaml', md: 'md', markdown: 'md'
  };

  // Настройка Marked.js для красивых блоков кода с кнопками Копировать / Скачать / Превью
  const renderer = new marked.Renderer();
  renderer.code = function(codeObjOrText, infostring) {
    let rawCode = typeof codeObjOrText === 'object' ? codeObjOrText.text : codeObjOrText;
    let lang = (typeof codeObjOrText === 'object' ? codeObjOrText.lang : infostring) || 'text';
    lang = lang.toLowerCase().trim();

    const id = 'code_' + (++codeBlockCounter);
    codeBlocksStore[id] = { code: rawCode, lang: lang };

    let highlighted = rawCode;
    try {
      if (hljs.getLanguage(lang)) {
        highlighted = hljs.highlight(rawCode, { language: lang }).value;
      } else {
        highlighted = hljs.highlightAuto(rawCode).value;
      }
    } catch (e) {}

    const isPreviewable = (lang === 'html' || lang === 'xml' || lang === 'svg');
    const previewBtn = isPreviewable
      ? `<button onclick="openPreview('${id}')" class="px-2 py-0.5 rounded bg-emerald-600/30 hover:bg-emerald-600/50 text-emerald-300 text-xs">▶️ Запустить</button>`
      : '';

    return `
      <div class="my-3 rounded-xl overflow-hidden border border-[#37393b] bg-[#1e1f20]">
        <div class="px-3 py-1.5 bg-[#282a2c] flex items-center justify-between text-xs text-gray-300 border-b border-[#37393b]">
          <span class="font-mono font-semibold uppercase">${lang}</span>
          <div class="flex items-center gap-2">
            ${previewBtn}
            <button onclick="copyCodeBlock('${id}', this)" class="px-2 py-0.5 rounded hover:bg-[#37393b]">📋 Копировать</button>
            <button onclick="downloadCodeBlock('${id}')" class="px-2 py-0.5 rounded hover:bg-[#37393b]">💾 Скачать файл</button>
          </div>
        </div>
        <pre class="p-3 overflow-x-auto text-xs font-mono leading-relaxed m-0"><code class="hljs">${highlighted}</code></pre>
      </div>
    `;
  };
  marked.setOptions({ renderer, breaks: true, gfm: true });

  // Инициализация при загрузке
  window.addEventListener('DOMContentLoaded', async () => {
    await loadModels(false);
    await loadChats();
    setupDragAndDrop();
    setupClipboardPaste();
  });

  async function loadModels(refresh = false) {
    const res = await fetch(`/api/models?refresh=${refresh}`);
    const data = await res.json();
    const sel = document.getElementById('modelSelect');
    const currentVal = sel.value || data.default;
    sel.innerHTML = '';
    data.models.forEach((m, idx) => {
      const opt = document.createElement('option');
      opt.value = m;
      opt.textContent = (idx < 3 ? '🔥 ' : '') + m;
      if (m === currentVal) opt.selected = true;
      sel.appendChild(opt);
    });
  }

  async function loadChats() {
    const res = await fetch('/api/chats');
    const data = await res.json();
    chats = data.chats;
    if (chats.length === 0) {
      await createNewChat();
    } else {
      renderChatList();
      await selectChat(chats[0].id);
    }
  }

  function renderChatList() {
    const list = document.getElementById('chatList');
    const q = document.getElementById('chatSearch').value.toLowerCase();
    list.innerHTML = '';

    chats.filter(c => c.title.toLowerCase().includes(q)).forEach(c => {
      const active = c.id === activeChatId;
      const div = document.createElement('div');
      div.className = `group flex items-center justify-between px-3 py-2.5 rounded-xl cursor-pointer text-sm transition ${
        active ? 'bg-[#282a2c] text-white font-medium' : 'text-gray-400 hover:bg-[#282a2c]/50 hover:text-gray-200'
      }`;
      div.onclick = () => selectChat(c.id);
      div.innerHTML = `
        <div class="truncate pr-2 flex items-center gap-2">
          <span>💬</span>
          <span class="truncate">${escapeHtml(c.title)}</span>
        </div>
        <button onclick="deleteChat(event, ${c.id})" title="Удалить чат" class="opacity-0 group-hover:opacity-100 hover:text-red-400 text-xs px-1">🗑</button>
      `;
      list.appendChild(div);
    });
  }

  async function createNewChat() {
    const model = document.getElementById('modelSelect').value || 'gemini-4-hydrogen';
    const sys = document.getElementById('systemPrompt').value || '';
    const fd = new FormData();
    fd.append('model', model);
    fd.append('system_prompt', sys);

    const res = await fetch('/api/chats', { method: 'POST', body: fd });
    const newChat = await res.json();
    chats.unshift(newChat);
    activeChatId = newChat.id;
    renderChatList();
    await selectChat(newChat.id);
  }

  async function selectChat(chatId) {
    activeChatId = chatId;
    renderChatList();
    const res = await fetch(`/api/chats/${chatId}/messages`);
    const data = await res.json();

    if (data.chat.model) {
      document.getElementById('modelSelect').value = data.chat.model;
    }
    document.getElementById('systemPrompt').value = data.chat.system_prompt || '';

    const container = document.getElementById('messagesContainer');
    container.innerHTML = '';

    if (data.messages.length === 0) {
      document.getElementById('welcomeScreen')?.classList.remove('hidden');
      container.innerHTML = document.getElementById('welcomeScreen') ? container.innerHTML : getWelcomeHtml();
    } else {
      data.messages.forEach(m => appendMessageBubble(m.role, m.content, m.attachments, m.images));
      scrollToBottom();
    }
  }

  async function navigateAdjacentChat(dir) {
    const idx = chats.findIndex(c => c.id === activeChatId);
    if (idx === -1) return;
    // Поскольку список отсортирован от новых к старым, "Прошлый" (-1) — это индекс + 1
    const targetIdx = idx - dir;
    if (targetIdx >= 0 && targetIdx < chats.length) {
      await selectChat(chats[targetIdx].id);
    }
  }

  async function deleteChat(e, chatId) {
    e.stopPropagation();
    await fetch(`/api/chats/${chatId}`, { method: 'DELETE' });
    chats = chats.filter(c => c.id !== chatId);
    if (chats.length === 0) {
      await createNewChat();
    } else if (activeChatId === chatId) {
      await selectChat(chats[0].id);
    } else {
      renderChatList();
    }
  }

  // Прикрепление файлов, Drag & Drop, Ctrl+V
  function handleFileSelect(e) {
    for (const file of e.target.files) {
      pendingFiles.push(file);
    }
    e.target.value = '';
    renderPendingFiles();
  }

  function renderPendingFiles() {
    const bar = document.getElementById('attachedFilesBar');
    if (pendingFiles.length === 0) {
      bar.classList.add('hidden');
      bar.innerHTML = '';
      return;
    }
    bar.classList.remove('hidden');
    bar.innerHTML = pendingFiles.map((f, i) => `
      <div class="flex items-center gap-2 bg-[#282a2c] border border-[#37393b] px-3 py-1.5 rounded-xl text-xs">
        <span>📄 ${escapeHtml(f.name)} <span class="text-gray-400">(${(f.size/1024).toFixed(1)} КБ)</span></span>
        <button onclick="removePendingFile(${i})" class="text-gray-400 hover:text-red-400">✕</button>
      </div>
    `).join('');
  }

  function removePendingFile(idx) {
    pendingFiles.splice(idx, 1);
    renderPendingFiles();
  }

  function setupDragAndDrop() {
    const dz = document.getElementById('dropZone');
    dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('ring-2', 'ring-blue-500'); });
    dz.addEventListener('dragleave', () => dz.classList.remove('ring-2', 'ring-blue-500'));
    dz.addEventListener('drop', e => {
      e.preventDefault();
      dz.classList.remove('ring-2', 'ring-blue-500');
      if (e.dataTransfer.files.length) {
        for (const f of e.dataTransfer.files) pendingFiles.push(f);
        renderPendingFiles();
      }
    });
  }

  function setupClipboardPaste() {
    window.addEventListener('paste', e => {
      if (e.clipboardData && e.clipboardData.files.length > 0) {
        for (const f of e.clipboardData.files) pendingFiles.push(f);
        renderPendingFiles();
      }
    });
  }

  // Запись голоса с микрофона
  async function toggleVoiceRecording() {
    const btn = document.getElementById('micBtn');
    const st = document.getElementById('micStatus');

    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.stop();
      btn.classList.remove('bg-red-600/30', 'text-red-300');
      st.textContent = 'Голос';
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recordedChunks = [];
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
      mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunks, { type: 'audio/webm' });
        const file = new File([blob], `voice_${Date.now()}.webm`, { type: 'audio/webm' });
        pendingFiles.push(file);
        renderPendingFiles();
        stream.getTracks().forEach(t => t.stop());
      };
      mediaRecorder.start();
      btn.classList.add('bg-red-600/30', 'text-red-300');
      st.textContent = 'Стоп ●';
    } catch (err) {
      alert('Нет доступа к микрофону: ' + err.message);
    }
  }

  // Отправка сообщения и потоковое чтение ответа
  async function sendMessage() {
    if (isGenerating) return;
    const input = document.getElementById('promptInput');
    const text = input.value.trim();
    if (!text && pendingFiles.length === 0) return;

    if (!activeChatId) await createNewChat();

    isGenerating = true;
    document.getElementById('sendBtn').disabled = true;

    const filesToSend = [...pendingFiles];
    const attPreview = filesToSend.map(f => ({ name: f.name, mime: f.type }));

    // Очищаем поле ввода
    input.value = '';
    input.style.height = 'auto';
    pendingFiles = [];
    renderPendingFiles();

    // Убираем приветственный экран, если он есть
    document.getElementById('welcomeScreen')?.remove();

    // Добавляем баббл пользователя
    appendMessageBubble('user', text, attPreview, []);

    // Создаем пустой баббл модели для стриминга
    const aiBubble = createStreamingBubble();
    scrollToBottom();

    const fd = new FormData();
    fd.append('chat_id', activeChatId);
    fd.append('prompt', text);
    fd.append('model', document.getElementById('modelSelect').value);
    fd.append('system_prompt', document.getElementById('systemPrompt').value);
    fd.append('temperature', document.getElementById('temperatureRange').value);
    fd.append('thinking_level', document.getElementById('thinkingLevel').value);
    fd.append('use_search', document.getElementById('searchToggle').checked);
    fd.append('use_code_exec', document.getElementById('codeExecToggle').checked);
    filesToSend.forEach(f => fd.append('files', f));

    let accumulatedText = '';
    let accumulatedImages = [];

    try {
      const response = await fetch('/api/chat/stream', { method: 'POST', body: fd });
      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.trim()) continue;
          const ev = JSON.parse(line);

          if (ev.type === 'status') {
            aiBubble.statusEl.textContent = ev.text;
            aiBubble.statusEl.classList.remove('hidden');
          } else if (ev.type === 'chunk') {
            aiBubble.statusEl.classList.add('hidden');
            accumulatedText += ev.text;
            aiBubble.contentEl.innerHTML = marked.parse(accumulatedText);
            scrollToBottom();
          } else if (ev.type === 'image') {
            aiBubble.statusEl.classList.add('hidden');
            accumulatedImages.push({ mime: ev.mime, data: ev.data });
            renderImagesInto(aiBubble.imagesEl, accumulatedImages);
            scrollToBottom();
          } else if (ev.type === 'done') {
            aiBubble.statusEl.classList.add('hidden');
            aiBubble.rawText = accumulatedText;
            aiBubble.actionsEl.classList.remove('hidden');
            if (ev.title) {
              const c = chats.find(x => x.id === activeChatId);
              if (c) { c.title = ev.title; renderChatList(); }
            }
          } else if (ev.type === 'error') {
            aiBubble.statusEl.classList.add('hidden');
            aiBubble.contentEl.innerHTML = `<div class="text-red-400 text-sm">❌ Ошибка: ${escapeHtml(ev.text)}</div>`;
          }
        }
      }
    } catch (err) {
      aiBubble.contentEl.innerHTML = `<div class="text-red-400 text-sm">❌ Сбой сети: ${escapeHtml(err.message)}</div>`;
    } finally {
      isGenerating = false;
      document.getElementById('sendBtn').disabled = false;
    }
  }

  function appendMessageBubble(role, content, attachments = [], images = []) {
    const container = document.getElementById('messagesContainer');
    const wrap = document.createElement('div');

    if (role === 'user') {
      const attHtml = (attachments || []).map(a =>
        `<span class="inline-flex items-center gap-1 bg-[#131314] border border-[#37393b] px-2.5 py-1 rounded-lg text-xs">📎 ${escapeHtml(a.name)}</span>`
      ).join(' ');

      wrap.className = 'flex justify-end';
      wrap.innerHTML = `
        <div class="max-w-2xl bg-[#282a2c] px-4 py-3 rounded-2xl rounded-tr-sm text-sm space-y-2">
          ${attHtml ? `<div class="flex flex-wrap gap-1.5">${attHtml}</div>` : ''}
          <div class="whitespace-pre-wrap">${escapeHtml(content)}</div>
        </div>
      `;
      container.appendChild(wrap);
    } else {
      const b = createStreamingBubble();
      b.statusEl.classList.add('hidden');
      b.rawText = content || '';
      b.contentEl.innerHTML = marked.parse(content || '');
      if (images && images.length) renderImagesInto(b.imagesEl, images);
      b.actionsEl.classList.remove('hidden');
    }
  }

  function createStreamingBubble() {
    const container = document.getElementById('messagesContainer');
    const wrap = document.createElement('div');
    wrap.className = 'flex gap-3 items-start';

    const bubbleObj = { rawText: '' };

    wrap.innerHTML = `
      <div class="w-8 h-8 rounded-full bg-gradient-to-tr from-blue-600 to-purple-600 flex items-center justify-center text-sm shrink-0 mt-1">✨</div>
      <div class="flex-1 min-w-0 space-y-2">
        <div class="status-line text-xs text-blue-400 animate-pulse">⏳ Подготовка...</div>
        <div class="images-box flex flex-wrap gap-3"></div>
        <div class="prose prose-invert max-w-none text-sm leading-relaxed content-box"></div>
        <div class="actions-bar hidden flex items-center gap-2 pt-1 text-xs text-gray-400">
          <button class="hover:text-white px-2 py-1 rounded hover:bg-[#282a2c] btn-copy">📋 Копировать</button>
          <button class="hover:text-white px-2 py-1 rounded hover:bg-[#282a2c] btn-md">📝 Скачать .md</button>
          <button class="hover:text-white px-2 py-1 rounded hover:bg-[#282a2c] btn-txt">📄 Скачать .txt</button>
          <button class="hover:text-white px-2 py-1 rounded hover:bg-[#282a2c] btn-tts">🔊 Озвучить</button>
        </div>
      </div>
    `;

    container.appendChild(wrap);
    bubbleObj.statusEl = wrap.querySelector('.status-line');
    bubbleObj.imagesEl = wrap.querySelector('.images-box');
    bubbleObj.contentEl = wrap.querySelector('.content-box');
    bubbleObj.actionsEl = wrap.querySelector('.actions-bar');

    wrap.querySelector('.btn-copy').onclick = () => navigator.clipboard.writeText(bubbleObj.rawText);
    wrap.querySelector('.btn-md').onclick = () => downloadTextFile(bubbleObj.rawText, `response_${Date.now()}.md`);
    wrap.querySelector('.btn-txt').onclick = () => downloadTextFile(bubbleObj.rawText, `response_${Date.now()}.txt`);
    wrap.querySelector('.btn-tts').onclick = () => speakText(bubbleObj.rawText);

    return bubbleObj;
  }

  function renderImagesInto(el, images) {
    el.innerHTML = images.map((img, i) => `
      <div class="relative group">
        <img src="data:${img.mime};base64,${img.data}" class="max-w-sm rounded-xl border border-[#37393b]" />
        <a href="data:${img.mime};base64,${img.data}" download="gemini_image_${i+1}.png"
          class="absolute bottom-2 right-2 bg-black/70 hover:bg-black text-white text-xs px-2.5 py-1 rounded-lg">💾 Скачать</a>
      </div>
    `).join('');
  }

  // Действия с кодом и экспорт
  function copyCodeBlock(id, btn) {
    const item = codeBlocksStore[id];
    if (!item) return;
    navigator.clipboard.writeText(item.code);
    const old = btn.textContent;
    btn.textContent = '✅ Скопировано';
    setTimeout(() => btn.textContent = old, 1500);
  }

  function downloadCodeBlock(id) {
    const item = codeBlocksStore[id];
    if (!item) return;
    const ext = langExtMap[item.lang] || 'txt';
    downloadTextFile(item.code, `snippet_${id}.${ext}`);
  }

  function openPreview(id) {
    const item = codeBlocksStore[id];
    if (!item) return;
    const modal = document.getElementById('previewModal');
    const frame = document.getElementById('previewFrame');
    frame.srcdoc = item.code;
    modal.classList.remove('hidden');
    modal.classList.add('flex');
  }

  function closePreviewModal() {
    const modal = document.getElementById('previewModal');
    modal.classList.add('hidden');
    modal.classList.remove('flex');
    document.getElementById('previewFrame').srcdoc = '';
  }

  function downloadTextFile(text, filename) {
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  async function exportCurrentChat() {
    if (!activeChatId) return;
    const res = await fetch(`/api/chats/${activeChatId}/messages`);
    const data = await res.json();
    const md = data.messages.map(m => `### ${m.role === 'user' ? '👤 Вы' : '✨ Gemini'}\n\n${m.content}\n`).join('\n---\n\n');
    downloadTextFile(md, `chat_${activeChatId}.md`);
  }

  function speakText(text) {
    if (!('speechSynthesis' in window)) return alert('Ваш браузер не поддерживает синтез речи');
    window.speechSynthesis.cancel();
    const clean = text.replace(/```[\s\S]*?```/g, ' [блок кода] ').replace(/[#*_`~]/g, '');
    const utter = new SpeechSynthesisUtterance(clean);
    utter.lang = 'ru-RU';
    window.speechSynthesis.speak(utter);
  }

  // Вспомогательные функции UI
  function quickPrompt(txt) {
    document.getElementById('promptInput').value = txt;
    sendMessage();
  }

  function autoResizeTextarea(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 192) + 'px';
  }

  function handleEnterKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('hidden');
  }

  function toggleRightDrawer() {
    const d = document.getElementById('rightDrawer');
    d.classList.toggle('hidden');
    d.classList.toggle('flex');
  }

  function onModelChange() {}

  function scrollToBottom() {
    const c = document.getElementById('messagesContainer');
    c.scrollTop = c.scrollHeight;
  }

  function escapeHtml(s) {
    return (s || '').replace(/[&<>"']/g, m => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[m]));
  }

  function getWelcomeHtml() {
    return `<div id="welcomeScreen" class="h-full flex flex-col items-center justify-center text-center py-16">
      <h1 class="text-4xl font-bold mb-3 gemini-gradient">Чем я могу помочь?</h1>
      <p class="text-gray-400 max-w-lg text-sm">Отправляйте текст, код, фото, видео, голосовые или PDF.</p>
    </div>`;
  }
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return HTML_PAGE


if __name__ == "__main__":
    init_db()
    print(f"🚀 Сайт запущен! Откройте в браузере: http://localhost:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)