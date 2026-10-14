import asyncio
import base64
import json
import logging
import socket
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

# os.environ["HF_HUB_DISABLE_XET"] = "1"

# Импортируем наш голосовой пайплайн
from voice_pipeline_optimization import VoiceAssistant


# ============================================================
# НАСТРОЙКИ
# ============================================================

HOST = "0.0.0.0"
PORT = 8000
WS_PATH = "/nils/v1/"

# Аудио: 16 кГц, моно, 16-бит PCM
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2                       # 16 бит = 2 байта
SERVER_AUDIO_CHUNK_BYTES = 2048        # размер чанка при отправке
CHUNK_DELAY_SEC = 0.03                 # пауза между чанками (чанк = 64 мс звука)

# Лимиты
MAX_AUDIO_BYTES = 40 * SAMPLE_RATE * SAMPLE_WIDTH  # ~40 секунд
MIN_AUDIO_BYTES = 8000                 # < 0,25 с — считаем "слишком коротко"

# Ответ, который не удалось отправить (ESP32 отключился во время обработки),
# хранится столько секунд и отправляется, когда устройство переподключится.
PENDING_TTL_SEC = 180

# Ping от сервера: таймаут с запасом, чтобы долгий синтез не рвал соединение
WS_PING_INTERVAL = 20.0
WS_PING_TIMEOUT = 60.0

# Фразы, которые устройство произносит вместо ответа
PHRASE_TOO_SHORT = "Слишком коротко, повторите, пожалуйста."
PHRASE_NOT_HEARD = "Не расслышал, повторите, пожалуйста."
PHRASE_ERROR = "Произошла ошибка, попробуйте ещё раз."

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Логи
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


# ============================================================
# ЛОГИРОВАНИЕ
# ============================================================

log_filename = LOG_DIR / f"server_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("Nils")


# ============================================================
# УТИЛИТЫ
# ============================================================

def get_local_ip() -> str:
    """Возвращает локальный IP для подключения ESP32."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ============================================================
# КЛАСС ПОДКЛЮЧЕНИЯ ESP32
# ============================================================

class ESP32Connection:
    """
    Хранит состояние одного подключённого ESP32:
    - буфер входящего аудио
    - флаг записи
    - метаданные сессии
    """

    def __init__(self, websocket: WebSocket, client_id: str):
        self.ws = websocket
        self.id = client_id
        # IP устройства: по нему ответ находит устройство после переподключения
        self.host = websocket.client.host if websocket.client else "unknown"
        self.audio_buffer = bytearray()
        self.is_recording = False
        self.connected_at = datetime.now()
        self.chunk_count = 0
        self.total_audio_bytes = 0
        self.processing_task: asyncio.Task | None = None

        logger.info(f"[{self.id}] Новое подключение с {self.host}")

    def websocket_open(self) -> bool:
        """Проверка, что сокет ещё открыт."""
        try:
            return (
                self.ws.client_state == WebSocketState.CONNECTED
                and self.ws.application_state == WebSocketState.CONNECTED
            )
        except Exception:
            return False

    # ---------- Запись ----------
    def start_recording(self):
        self.is_recording = True
        self.audio_buffer = bytearray()
        self.chunk_count = 0
        self.total_audio_bytes = 0
        logger.info(f"[{self.id}] Начало записи")

    def add_audio(self, chunk: bytes):
        if not self.is_recording:
            logger.warning(f"[{self.id}] Чанк без активной записи — игнор")
            return

        if self.total_audio_bytes + len(chunk) > MAX_AUDIO_BYTES:
            self.audio_buffer = bytearray()
            self.is_recording = False
            raise RuntimeError(
                f"Превышен лимит входящего аудио "
                f"({self.total_audio_bytes + len(chunk)} байт)"
            )

        self.chunk_count += 1
        self.total_audio_bytes += len(chunk)
        self.audio_buffer.extend(chunk)

        if self.chunk_count % 10 == 0:
            logger.info(
                f"[{self.id}] Получено {self.chunk_count} чанков, "
                f"всего {len(self.audio_buffer)} байт"
            )

    def finish_recording(self) -> bytes:
        if not self.is_recording and not self.audio_buffer:
            raise RuntimeError("Нет активной записи")

        audio = bytes(self.audio_buffer)
        self.audio_buffer = bytearray()
        self.is_recording = False
        logger.info(
            f"[{self.id}] Запись завершена: "
            f"{self.chunk_count} чанков, {len(audio)} байт"
        )
        return audio

    # ---------- Отправка ----------
    async def send_json(self, payload: dict) -> bool:
        if not self.websocket_open():
            logger.warning(f"[{self.id}] Попытка отправки в закрытый сокет ({payload.get('type')})")
            return False
        try:
            await self.ws.send_json(payload)
            return True
        except Exception as e:
            logger.error(f"[{self.id}] Ошибка отправки JSON: {e}")
            return False

    async def send_text(self, text: str) -> bool:
        return await self.send_json({"type": "text_response", "text": text})

    async def send_audio(self, audio_bytes: bytes) -> bool:
        """Отправляет PCM16 аудио чанками. True — если дошло до audio_response_end."""
        if not audio_bytes:
            logger.error(f"[{self.id}] Пустое аудио для отправки")
            return False

        total_bytes = len(audio_bytes)
        total_chunks = (total_bytes + SERVER_AUDIO_CHUNK_BYTES - 1) // SERVER_AUDIO_CHUNK_BYTES

        logger.info(
            f"[{self.id}] Отправка аудио: {total_bytes} байт, "
            f"{total_chunks} чанков"
        )

        # 1. Стартовое сообщение
        if not await self.send_json({
            "type": "audio_response_start",
            "format": "pcm16",
            "sample_rate": SAMPLE_RATE,
            "total_bytes": total_bytes,
            "chunk_bytes": SERVER_AUDIO_CHUNK_BYTES,
            "total_chunks": total_chunks,
        }):
            return False

        # 2. Чанки
        sent = 0
        for index in range(total_chunks):
            chunk = audio_bytes[sent:sent + SERVER_AUDIO_CHUNK_BYTES]
            sent += len(chunk)

            if not await self.send_json({
                "type": "audio_response_chunk",
                "index": index,
                "data": base64.b64encode(chunk).decode("utf-8"),
            }):
                logger.warning(f"[{self.id}] Прервана отправка на чанке {index}/{total_chunks}")
                return False

            # Небольшая задержка, чтобы ESP32 успевал воспроизводить
            await asyncio.sleep(CHUNK_DELAY_SEC)

        # 3. Финал
        if not await self.send_json({
            "type": "audio_response_end",
            "total_chunks": total_chunks,
        }):
            return False
        logger.info(f"[{self.id}] Отправка аудио завершена")
        return True


# ============================================================
# ГЛОБАЛЬНОЕ СОСТОЯНИЕ
# ============================================================

app = FastAPI(title="Nils Voice Server")

active_connections: dict[str, ESP32Connection] = {}

# Неотправленные ответы по IP устройства: {"time", "text", "audio"}
pending_responses: dict[str, dict] = {}

# Модели (GPU, история диалога) не рассчитаны на параллельные запросы
pipeline_lock = asyncio.Lock()

# Ссылки на фоновые задачи, чтобы их не собрал сборщик мусора
background_tasks: set[asyncio.Task] = set()

# Голосовой ассистент — создаётся один раз при старте сервера
assistant: VoiceAssistant | None = None


def spawn(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    return task


def live_connection(host: str) -> ESP32Connection | None:
    """Самое свежее открытое подключение с этого IP (после переподключения — новое)."""
    candidates = [c for c in active_connections.values() if c.host == host and c.websocket_open()]
    return max(candidates, key=lambda c: c.connected_at, default=None)


# ============================================================
# ДОСТАВКА ОТВЕТА
# ============================================================

async def deliver(conn: ESP32Connection, text: str, audio: bytes):
    """
    Отправляет ответ на текущее подключение устройства. Если ESP32 за время обработки
    отключился и ещё не вернулся, ответ откладывается до переподключения.
    """
    target = live_connection(conn.host)
    if target is not None:
        await target.send_text(text)
        if await target.send_audio(audio):
            return
    logger.warning(
        f"[{conn.id}] Ответ НЕ доставлен (устройство {conn.host} не на связи) — "
        f"отправлю при переподключении в течение {PENDING_TTL_SEC} с"
    )
    pending_responses[conn.host] = {"time": time.time(), "text": text, "audio": audio}


async def deliver_pending(conn: ESP32Connection):
    pending = pending_responses.pop(conn.host, None)
    if pending is None:
        return
    age = time.time() - pending["time"]
    if age > PENDING_TTL_SEC:
        logger.info(f"[{conn.id}] Отложенный ответ устарел ({age:.0f} с) — отбрасываю")
        return
    logger.info(f"[{conn.id}] Отправляю отложенный ответ ({age:.0f} с назад)")
    await conn.send_text(pending["text"])
    if not await conn.send_audio(pending["audio"]):
        pending_responses[conn.host] = pending


async def synthesize_pcm(text: str) -> bytes:
    """TTS → PCM16 байты 16 кГц."""
    return await asyncio.to_thread(assistant.tts_pcm16, text)


class StreamingReply:
    """
    Отправляет ответ по мере синтеза: audio_response_start → чанки каждого
    предложения → audio_response_end. Прошивка играет чанки сразу, так что паузы
    между предложениями (пока синтезируется следующее) — просто тишина.
    Если устройство отключилось посреди ответа, весь ответ целиком доставляется
    через deliver() — на новое подключение или после переподключения.
    """

    def __init__(self, conn: ESP32Connection):
        self.conn = conn
        self.target = live_connection(conn.host)
        self.failed = self.target is None
        self.started = False
        self.index = 0
        self.audio = bytearray()
        self.first_audio_at: float | None = None

    async def add(self, pcm: bytes):
        self.audio += pcm
        if self.failed:
            return
        if not self.started:
            if not await self.target.send_json({
                "type": "audio_response_start",
                "format": "pcm16",
                "sample_rate": SAMPLE_RATE,
                "streaming": True,
            }):
                self.failed = True
                return
            self.started = True
            self.first_audio_at = time.time()
        for offset in range(0, len(pcm), SERVER_AUDIO_CHUNK_BYTES):
            if not await self.target.send_json({
                "type": "audio_response_chunk",
                "index": self.index,
                "data": base64.b64encode(pcm[offset:offset + SERVER_AUDIO_CHUNK_BYTES]).decode("utf-8"),
            }):
                logger.warning(f"[{self.conn.id}] Потоковая отправка прервана на чанке {self.index}")
                self.failed = True
                return
            self.index += 1
            await asyncio.sleep(CHUNK_DELAY_SEC)

    async def finish(self, text: str):
        if not self.failed and self.started:
            if await self.target.send_json({"type": "audio_response_end", "total_chunks": self.index}):
                await self.target.send_text(text)
                logger.info(f"[{self.conn.id}] Потоковый ответ доставлен ({self.index} чанков)")
                return
        if self.audio:
            await deliver(self.conn, text, bytes(self.audio))
        else:
            await speak(self.conn, PHRASE_ERROR)

    async def abort(self):
        """Закрыть начатый ответ, чтобы устройство не ждало до своего таймаута."""
        if self.started and not self.failed:
            await self.target.send_json({"type": "audio_response_end", "total_chunks": self.index})


async def speak(conn: ESP32Connection, text: str):
    """
    Озвучивает короткую служебную фразу. Устройство ждёт аудио-ответ, поэтому
    без него оно молчало бы до своего таймаута; при сбое TTS хотя бы
    отправляем audio_response_end, чтобы ESP32 вернулся к ожиданию.
    """
    try:
        audio = await synthesize_pcm(text)
        await deliver(conn, text, audio)
    except Exception as e:
        logger.error(f"[{conn.id}] Не удалось озвучить «{text}»: {e}")
        target = live_connection(conn.host)
        if target is not None:
            await target.send_text(text)
            await target.send_json({"type": "audio_response_end", "total_chunks": 0})


# ============================================================
# ПОТОКОВЫЙ ОТВЕТ: LLM → TTS → ESP32 одновременно
# ============================================================

async def stream_answer(conn: ESP32Connection, user_text: str):
    """
    Три стадии работают параллельно:
      1. LLM (поток) пишет ответ и отдаёт законченные предложения;
      2. Silero (поток) озвучивает каждое предложение, как только оно пришло;
      3. отправка шлёт готовое аудио на ESP32, пока синтезируется следующее.
    Первый звук появляется после первого предложения, а не после всего ответа.
    Возвращает (текст ответа, StreamingReply, время LLM, суммарное время TTS).
    """
    loop = asyncio.get_running_loop()
    sentences: asyncio.Queue = asyncio.Queue()
    audio_queue: asyncio.Queue = asyncio.Queue()
    reply = StreamingReply(conn)
    spoken: list[str] = []
    timing = {"llm": 0.0, "tts": 0.0}
    t0 = time.time()

    def produce_sentences():
        try:
            for sentence in assistant.ask_llm_stream(user_text):
                loop.call_soon_threadsafe(sentences.put_nowait, sentence)
        finally:
            timing["llm"] = time.time() - t0
            loop.call_soon_threadsafe(sentences.put_nowait, None)

    async def synthesize_sentences():
        try:
            while (sentence := await sentences.get()) is not None:
                spoken.append(sentence)
                t = time.time()
                try:
                    pcm = await synthesize_pcm(sentence)
                except Exception as e:
                    logger.error(f"[{conn.id}] TTS не смог озвучить предложение, пропускаю: {sentence!r} ({e})")
                    continue
                timing["tts"] += time.time() - t
                await audio_queue.put(pcm)
        finally:
            await audio_queue.put(None)

    async def send_audio():
        while (pcm := await audio_queue.get()) is not None:
            await reply.add(pcm)

    try:
        await asyncio.gather(asyncio.to_thread(produce_sentences), synthesize_sentences(), send_audio())
    except Exception:
        await reply.abort()
        raise

    answer = " ".join(spoken)
    await reply.finish(answer)
    return answer, reply, timing["llm"], timing["tts"]


# ============================================================
# ОБРАБОТКА АУДИО (в отдельной задаче)
# ============================================================

async def process_audio(conn: ESP32Connection):
    """
    Полный цикл: PCM-байты → STT → LLM → TTS → отправка обратно.
    Запускается как фоновая задача и НЕ отменяется при отключении ESP32:
    готовый ответ будет доставлен после переподключения.
    """
    t_start = time.time()
    try:
        audio = conn.finish_recording()

        async with pipeline_lock:
            if len(audio) < MIN_AUDIO_BYTES:
                logger.warning(f"[{conn.id}] Слишком короткая запись: {len(audio)} байт")
                await speak(conn, PHRASE_TOO_SHORT)
                return

            logger.info(f"[{conn.id}] Начинаю обработку: {len(audio)} байт "
                        f"({len(audio) / (SAMPLE_RATE * SAMPLE_WIDTH):.1f} с)")

            # Сохраняем входящее аудио для отладки и как вход для Whisper
            audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            debug_wav = OUTPUT_DIR / f"{conn.id}_input.wav"
            sf.write(str(debug_wav), audio_np, SAMPLE_RATE, subtype="PCM_16")

            # ---------- STT ----------
            t0 = time.time()
            user_text = await asyncio.to_thread(assistant.transcribe, debug_wav)
            t_stt = time.time() - t0
            if not user_text:
                logger.warning(f"[{conn.id}] STT не распознал речь")
                await speak(conn, PHRASE_NOT_HEARD)
                return

            target = live_connection(conn.host)
            if target:
                await target.send_text(f"Вы сказали: {user_text}")

            # ---------- LLM → TTS → отправка, потоково по предложениям ----------
            answer, reply, t_llm, t_tts = await stream_answer(conn, user_text)

            pcm = np.frombuffer(bytes(reply.audio), dtype=np.int16)
            if len(pcm):
                sf.write(str(OUTPUT_DIR / f"{conn.id}_response.wav"), pcm, SAMPLE_RATE, subtype="PCM_16")

        first = (f"{reply.first_audio_at - t_start:.1f} с" if reply.first_audio_at else "—")
        logger.info(
            f"[{conn.id}] Полный цикл за {time.time() - t_start:.1f} с, первый звук через {first} "
            f"(STT {t_stt:.1f} с, LLM {t_llm:.1f} с, TTS суммарно {t_tts:.1f} с, "
            f"аудио {len(pcm) / SAMPLE_RATE:.1f} с)"
        )

    except Exception as e:
        logger.exception(f"[{conn.id}] Ошибка в process_audio: {e}")
        await speak(conn, PHRASE_ERROR)
    finally:
        conn.processing_task = None


# ============================================================
# WEBSOCKET ENDPOINT
# ============================================================

@app.websocket(WS_PATH)
async def websocket_endpoint(websocket: WebSocket):
    client_id = str(uuid.uuid4())[:8]
    conn = None
    try:
        await websocket.accept()
        logger.info(f"Новое WebSocket-подключение: {client_id}")

        conn = ESP32Connection(websocket, client_id)
        active_connections[client_id] = conn

        # Приветственное сообщение
        await websocket.send_json({
            "type": "hello",
            "message": "Connected to Nils Voice Server",
            "client_id": client_id,
        })

        # Ответ, который не дошёл до прошлого подключения этого устройства
        if conn.host in pending_responses:
            spawn(deliver_pending(conn))

        # Основной цикл приёма сообщений
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError as e:
                logger.error(f"[{client_id}] Ошибка парсинга JSON: {e}")
                continue

            msg_type = message.get("type")
            logger.debug(f"[{client_id}] Получено: {msg_type}")

            # ---------- PING ----------
            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            # ---------- START ----------
            elif msg_type == "start_recording":
                conn.start_recording()

            # ---------- AUDIO CHUNK ----------
            elif msg_type == "audio_chunk":
                audio_data = message.get("data")
                index = message.get("index", -1)
                if not audio_data or not isinstance(audio_data, str):
                    logger.error(f"[{client_id}] Некорректный чанк #{index}")
                    continue
                try:
                    chunk = base64.b64decode(audio_data, validate=True)
                    conn.add_audio(chunk)
                except Exception as e:
                    logger.error(f"[{client_id}] Ошибка декодирования чанка #{index}: {e}")

            # ---------- STOP ----------
            elif msg_type == "stop_recording":
                logger.info(f"[{client_id}] Получена команда stop_recording")
                if not conn.is_recording:
                    logger.warning(f"[{client_id}] stop_recording без активной записи")
                    continue
                if conn.processing_task and not conn.processing_task.done():
                    logger.warning(f"[{client_id}] Обработка уже идёт")
                    continue
                # Запускаем обработку в фоне
                conn.processing_task = spawn(process_audio(conn))

            else:
                logger.warning(f"[{client_id}] Неизвестный тип: {msg_type}")

    except WebSocketDisconnect:
        logger.info(f"ESP32 отключился: {client_id}")
    except Exception as e:
        logger.exception(f"Ошибка соединения {client_id}: {e}")
    finally:
        # Обработку НЕ отменяем: раньше именно здесь терялся ответ — WAV уже сохранялся,
        # а отправка отменялась. Теперь ответ дойдёт до переподключившегося устройства.
        if conn and conn.processing_task and not conn.processing_task.done():
            logger.info(f"[{conn.id}] Обработка продолжается, ответ будет доставлен после переподключения")
        if conn and conn.id in active_connections:
            del active_connections[conn.id]
            logger.info(f"[{conn.id}] Соединение закрыто")


# ============================================================
# REST-ЭНДПОИНТЫ (для проверки)
# ============================================================

@app.get("/")
async def root():
    return {
        "name": "Nils Voice Server",
        "status": "running",
        "connections": len(active_connections),
        "pending_responses": len(pending_responses),
        "ws_path": WS_PATH,
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "time": datetime.now().isoformat(),
        "connections": len(active_connections),
    }


@app.get("/connections")
async def list_connections():
    return {
        "total": len(active_connections),
        "devices": [
            {
                "id": c.id,
                "host": c.host,
                "connected": c.connected_at.isoformat(),
                "recording": c.is_recording,
                "processing": bool(c.processing_task and not c.processing_task.done()),
                "buffered_bytes": len(c.audio_buffer),
            }
            for c in active_connections.values()
        ],
    }


# ============================================================
# ЗАПУСК
# ============================================================

@app.on_event("startup")
async def on_startup():
    """Загружаем модели при старте сервера."""
    global assistant
    logger.info("=" * 60)
    logger.info("ЗАПУСК СЕРВЕРА НИЛЬС")
    logger.info("=" * 60)
    logger.info("Загрузка моделей (Whisper + Silero + прогрев LLM)...")
    assistant = VoiceAssistant(preload=True)
    logger.info("Сервер готов к работе")
    logger.info("=" * 60)


if __name__ == "__main__":
    local_ip = get_local_ip()
    print("\n" + "=" * 60)
    print("Нильс бета — ГОЛОСОВОЙ СЕРВЕР")
    print("=" * 60)
    print(f"Локальный IP:      {local_ip}")
    print(f"WebSocket URL:     ws://{local_ip}:{PORT}{WS_PATH}")
    print(f"REST API:          http://{local_ip}:{PORT}/")
    print(f"Логи:              {log_filename}")
    print("=" * 60 + "\n")

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
        reload=False,
        ws_ping_interval=WS_PING_INTERVAL,
        ws_ping_timeout=WS_PING_TIMEOUT,
    )
