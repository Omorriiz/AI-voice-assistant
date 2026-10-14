import os
import re
import time
import argparse
from math import gcd
from pathlib import Path
from collections import deque

os.environ["HF_HUB_DISABLE_XET"] = "1"

import numpy as np
import scipy.signal
import torch
import soundfile as sf
from faster_whisper import WhisperModel
from ollama import chat

try:
    from num2words import num2words  
except ImportError:
    num2words = None


# ============================================================
# CONFIG
# ============================================================

# --- STT ---
WHISPER_MODEL_DIR = r"C:/Users/yumit/Desktop/Degree/server/model_faster_Whisper"
WHISPER_LANG      = "ru"
WHISPER_DEVICE    = "cuda"          # "cuda" или "cpu"
WHISPER_COMPUTE   = "float16"       # "float16" для GPU, "int8" для CPU
WHISPER_BEAM      = 5               # 1=быстро, 5=точно

# --- CPU ---
def _physical_cores() -> int:
    """Физические ядра: на CPU гиперпотоки замедляют генерацию llama.cpp, а не ускоряют."""
    try:
        import psutil
        cores = psutil.cpu_count(logical=False)
        if cores:
            return cores
    except ImportError:
        pass
    return max(2, (os.cpu_count() or 4) // 2)


CPU_CORES = _physical_cores()
# LLM и Silero работают одновременно (ответ озвучивается по предложениям, пока LLM пишет
# дальше), поэтому ядра делятся между ними, а не отдаются целиком каждому.
TTS_THREADS       = 2 if CPU_CORES <= 6 else 4

# --- LLM ---
# Скорость на CPU определяется в первую очередь размером модели: 7-8B в q4_K_M примерно
# в полтора раза быстрее 12B, квантизация q3_K_M — ещё на ~15-20% быстрее (ценой качества).
OLLAMA_MODEL      = "OxW/Vikhr-Nemo-12B-Instruct-R-21-09-24:q4_K_M"
OLLAMA_THREADS    = max(2, CPU_CORES - TTS_THREADS)  # или задайте число вручную
# Один размер контекста для всех режимов: при смене num_ctx Ollama перезагружает модель.
# 2048 хватает на системный промпт + 5 пар истории; меньше контекст — быстрее на CPU.
OLLAMA_NUM_CTX    = 2048
# Держать модель в памяти всегда: по умолчанию Ollama выгружает её через 5 минут простоя,
# и первый запрос после паузы ждал перезагрузку модели.
OLLAMA_KEEP_ALIVE = -1

# --- TTS ---
SILERO_SPEAKER    = "xenia"         # aidar, baya, kseniya, xenia, eugene
SILERO_SR         = 48000           # частота Silero
TARGET_SR         = 16000           # частота для ESP32
TTS_MAX_CHARS     = 600             # Silero падает на слишком длинном тексте — синтезируем по частям
TTS_PAUSE_SEC     = 0.15            # пауза между частями

# --- Потоковая генерация ---
# Ответ отдаётся на озвучку по предложениям; совсем короткие фразы склеиваются со следующими,
# чтобы не синтезировать по одному слову.
MIN_SENTENCE_CHARS = 25
SENTENCE_END_RE    = re.compile(r"[.!?…]+[»\"')]*\s+")

# --- Фразы-галлюцинации Whisper на тишине/шуме (такой «текст» считается пустым) ---
WHISPER_HALLUCINATIONS = [
    "продолжение следует", "субтитры", "редактор субтитров", "корректор",
    "спасибо за просмотр", "подписывайтесь на канал", "ставьте лайки",
    "dimatorzok", "amara.org",
]

# --- История ---
MAX_HISTORY_PAIRS = 5               # храним последние 5 пар (user+assistant)

# --- Общие правила вывода (для синтеза речи) ---
OUTPUT_RULES = (
    " Пиши только русскими буквами, без цифр и латиницы. "
    "Все числа, даты, время, проценты, денежные суммы и единицы измерения пиши словами "
    "в правильном падеже (например: «двадцать пять процентов», а не «25%»; "
    "«в две тысячи двадцать шестом году», а не «в 2026 году»). "
    "Все иностранные слова, названия, бренды и термины пиши русской транслитерацией так, "
    "как они произносятся по-русски (например: Bitcoin — Биткоин, iPhone — Айфон, "
    "Google — Гугл, Python — Пайтон, AI — Эй-Ай)."
)

# --- Промпты под разные режимы ответа ---
SHORT_SYSTEM_PROMPT = (
    "Ты голосовой ассистент. Отвечай ОДНИМ-ДВУМЯ короткими предложениями, "
    "максимум 15 слов. Без markdown, без эмодзи, без списков." + OUTPUT_RULES
)

MEDIUM_SYSTEM_PROMPT = (
    "Ты голосовой ассистент. Отвечай тремя-пятью предложениями, "
    "максимум 50 слов. Без markdown, без эмодзи, без списков." + OUTPUT_RULES
)

LONG_SYSTEM_PROMPT = (
    "Ты голосовой ассистент. Отвечай развёрнуто, но структурированно: "
    "шесть-десять предложений, максимум 150 слов. Без markdown, без эмодзи, "
    "без списков." + OUTPUT_RULES
)

# --- Триггеры для выбора режима ---
LONG_TRIGGERS = [
    "подробно", "подробнее", "поподробнее", "развёрнуто", "развернуто",
    "расскажи", "объясни", "в деталях", "детально",
    "почему", "как работает", "в чём разница", "в чем разница",
    "сравни", "лекция", "история", "теория",
]

MEDIUM_TRIGGERS = [
    "что такое", "кто такой", "кто такая", "опиши", "назови",
    "перечисли", "какие", "чем отличается",
]

OUTPUT_DIR = Path(r"C:\Users\yumit\Desktop\Degree\server\output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# НОРМАЛИЗАЦИЯ ТЕКСТА ДЛЯ TTS
# Страховка на случай, если LLM всё же выдаст цифры или латиницу:
# Silero читает только кириллицу.
# ============================================================

# Известные термины: точная транслитерация (ключи — в нижнем регистре)
TERMS = {
    "bitcoin": "биткоин", "ethereum": "эфириум", "blockchain": "блокчейн",
    "iphone": "айфон", "ipad": "айпад", "android": "андроид", "apple": "эппл",
    "google": "гугл", "youtube": "ютуб", "telegram": "телеграм", "whatsapp": "ватсап",
    "instagram": "инстаграм", "facebook": "фейсбук", "twitter": "твиттер",
    "microsoft": "майкрософт", "windows": "виндоус", "linux": "линукс",
    "python": "пайтон", "javascript": "джаваскрипт", "java": "джава",
    "wi-fi": "вай-фай", "wifi": "вайфай", "bluetooth": "блютус",
    "chatgpt": "чат джи-пи-ти", "openai": "оупен эй-ай", "nvidia": "энвидиа",
    "intel": "интел", "samsung": "самсунг", "amazon": "амазон", "tesla": "тесла",
    "netflix": "нетфликс", "spotify": "спотифай", "zoom": "зум", "skype": "скайп",
    "email": "имейл", "e-mail": "имейл", "online": "онлайн", "offline": "офлайн",
    "internet": "интернет", "ok": "окей", "okay": "окей", "hello": "хэллоу",
    "esp": "и-эс-пи", "esp32": "и-эс-пи тридцать два",
}

# Названия английских букв для аббревиатур (AI → эй-ай, USB → ю-эс-би)
LETTER_NAMES = {
    "a": "эй", "b": "би", "c": "си", "d": "ди", "e": "и", "f": "эф", "g": "джи",
    "h": "эйч", "i": "ай", "j": "джей", "k": "кей", "l": "эл", "m": "эм", "n": "эн",
    "o": "оу", "p": "пи", "q": "кью", "r": "ар", "s": "эс", "t": "ти", "u": "ю",
    "v": "ви", "w": "дабл-ю", "x": "экс", "y": "уай", "z": "зед",
}

# Приблизительная транслитерация «по-английски» для остальных слов
TRANSLIT_MULTI = [
    ("sch", "ш"), ("tion", "шн"), ("sh", "ш"), ("ch", "ч"), ("th", "т"), ("ph", "ф"),
    ("ck", "к"), ("qu", "кв"), ("oo", "у"), ("ee", "и"), ("ea", "и"), ("ou", "ау"),
    ("ay", "эй"), ("ai", "эй"), ("oa", "о"), ("kh", "х"), ("zh", "ж"), ("ts", "ц"),
    ("ya", "я"), ("yu", "ю"), ("yo", "йо"), ("wh", "в"), ("gh", "г"), ("x", "кс"),
]
TRANSLIT_SINGLE = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х",
    "i": "и", "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п",
    "q": "к", "r": "р", "s": "с", "t": "т", "u": "у", "v": "в", "w": "в", "y": "и",
    "z": "з",
}

LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-']*")
NUMBER_RE = re.compile(r"(?<![\w])(-?\d{1,3}(?:[ \u00a0]\d{3})+|-?\d+)(?:[.,](\d+))?(?![\w])")
SYMBOLS = {"%": " процентов", "$": " долларов", "€": " евро", "₽": " рублей", "°": " градусов",
           "&": " и ", "+": " плюс ", "=": " равно "}


def _translit_word(word: str) -> str:
    lower = word.lower()
    out, i = [], 0
    while i < len(lower):
        for src, dst in TRANSLIT_MULTI:
            if lower.startswith(src, i):
                out.append(dst)
                i += len(src)
                break
        else:
            ch = lower[i]
            nxt = lower[i + 1] if i + 1 < len(lower) else ""
            if ch == "c" and nxt in "eiy":
                out.append("с")
            elif ch == "e" and i == 0:
                out.append("э")
            elif ch == "y" and i == 0:
                out.append("й")
            elif ch == "e" and i == len(lower) - 1 and len(lower) > 3:
                pass  # немая e на конце: "line" → "лайн"-подобное "лин"
            else:
                out.append(TRANSLIT_SINGLE.get(ch, ch))
            i += 1
    return "".join(out)


def _latin_to_russian(match: re.Match) -> str:
    word = match.group(0)
    key = word.lower()
    if key in TERMS:
        result = TERMS[key]
    elif word.isupper() and 2 <= len(word) <= 5 and word.isalpha():
        result = "-".join(LETTER_NAMES[c] for c in key)  # аббревиатура
    else:
        result = _translit_word(word)
    return result.capitalize() if word[0].isupper() else result


def _number_to_words(match: re.Match) -> str:
    if num2words is None:
        return match.group(0)
    integer = match.group(1).replace(" ", "").replace("\u00a0", "")
    fraction = match.group(2)
    try:
        if fraction:
            return " " + num2words(float(f"{integer}.{fraction}"), lang="ru") + " "
        return " " + num2words(int(integer), lang="ru") + " "
    except Exception:
        return match.group(0)


def normalize_for_tts(text: str) -> str:
    """Числа → слова, латиница → кириллица, лишние символы → пробелы."""
    text = re.sub(r"[*_#`>\[\]]", " ", text)                  # остатки markdown
    text = re.sub(r"(\d{1,2}):(\d{2})", r"\1 \2", text)       # время 10:30 → 10 30
    for sym, word in SYMBOLS.items():
        text = text.replace(sym, word)
    text = NUMBER_RE.sub(_number_to_words, text)
    text = LATIN_WORD_RE.sub(_latin_to_russian, text)
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"\s+([,.!?;:…])", r"\1", text).strip()


def split_for_tts(text: str, max_chars: int = TTS_MAX_CHARS) -> list[str]:
    """Режет текст по предложениям на куски не длиннее max_chars."""
    sentences = re.split(r"(?<=[.!?…])\s+", text)
    parts, current = [], ""
    for sentence in sentences:
        while len(sentence) > max_chars:  # очень длинное предложение — режем по запятым/пробелам
            cut = max(sentence.rfind(",", 0, max_chars), sentence.rfind(" ", 0, max_chars))
            cut = cut if cut > 0 else max_chars
            parts.append(sentence[:cut + 1].strip())
            sentence = sentence[cut + 1:].strip()
        if current and len(current) + 1 + len(sentence) > max_chars:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    return [p for p in parts if re.search(r"[а-яА-ЯёЁ]", p)]


# ============================================================
# КЛАСС-ОБЁРТКА
# ============================================================

class VoiceAssistant:
    def __init__(self, preload: bool = True, beam_size: int = WHISPER_BEAM):
        self.whisper = None
        self.tts_model = None
        self.beam_size = beam_size

        # История диалога (без системного промпта: он добавляется при каждом запросе,
        # иначе deque с maxlen вытеснял бы его первым)
        self.history = deque(maxlen=MAX_HISTORY_PAIRS * 2)

        if num2words is None:
            print("[TTS] ВНИМАНИЕ: num2words не установлен (pip install num2words) — "
                  "цифры в ответе не будут превращаться в слова")

        if preload:
            print("=" * 60)
            print("ПРЕДЗАГРУЗКА МОДЕЛЕЙ")
            print("=" * 60)
            self.load_whisper()
            self.load_tts()
            self._warmup_llm()
            print("=" * 60 + "\n")

    # ---------- STT ----------
    def load_whisper(self):
        if self.whisper is None:
            print(f"[STT] Загружаю Faster-Whisper ({WHISPER_DEVICE}/{WHISPER_COMPUTE})...")
            t0 = time.time()
            self.whisper = WhisperModel(
                WHISPER_MODEL_DIR,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE,
            )
            print(f"[STT] Готово за {time.time() - t0:.1f} сек")
        return self.whisper

    def transcribe(self, wav_path: Path) -> str:
        self.load_whisper()
        print(f"[STT] Распознаю (beam_size={self.beam_size}): {wav_path}")
        t0 = time.time()
        segments, _info = self.whisper.transcribe(
            str(wav_path),
            language=WHISPER_LANG,
            task="transcribe",
            beam_size=self.beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        text = " ".join(s.text.strip() for s in segments if s.text.strip()).strip()
        print(f"[STT] Распознано за {time.time() - t0:.2f} сек: {text!r}")

        lowered = text.lower()
        if any(h in lowered for h in WHISPER_HALLUCINATIONS) or not re.search(r"[а-яА-ЯёЁa-zA-Z]{2}", text):
            print(f"[STT] Похоже на галлюцинацию Whisper на тишине — игнорирую: {text!r}")
            return ""
        return text

    # ---------- LLM ----------
    def _warmup_llm(self):
        """Прогрев LLM: короткий запрос, чтобы модель загрузилась в память."""
        print(f"[LLM] Прогрев модели {OLLAMA_MODEL}...")
        t0 = time.time()
        try:
            chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": "привет"}],
                # Те же num_ctx/num_thread, что и в рабочих запросах, иначе Ollama перезагрузит модель
                options={"num_predict": 1, "num_ctx": OLLAMA_NUM_CTX, "num_thread": OLLAMA_THREADS},
                keep_alive=OLLAMA_KEEP_ALIVE,
            )
            print(f"[LLM] Прогрев за {time.time() - t0:.1f} сек")
        except Exception as e:
            print(f"[LLM] Прогрев не удался: {e}")

    def _choose_mode(self, user_text: str) -> tuple[str, int, str]:
        """
        Определяет режим ответа по запросу пользователя.
        Возвращает: (system_prompt, num_predict, mode_name)
        """
        text_lower = user_text.lower()

        if any(t in text_lower for t in LONG_TRIGGERS):
            return LONG_SYSTEM_PROMPT, 600, "длинный"
        elif any(t in text_lower for t in MEDIUM_TRIGGERS):
            return MEDIUM_SYSTEM_PROMPT, 250, "средний"
        else:
            return SHORT_SYSTEM_PROMPT, 80, "короткий"

    @staticmethod
    def _split_point(buffer: str) -> int | None:
        """Позиция конца первого законченного предложения не короче MIN_SENTENCE_CHARS."""
        for m in SENTENCE_END_RE.finditer(buffer):
            if m.end() >= MIN_SENTENCE_CHARS:
                return m.end()
        return None

    def ask_llm_stream(self, user_text: str, reset_history: bool = False):
        """
        Генерирует ответ потоково и отдаёт его по предложениям, как только каждое готово.
        История обновляется, когда генератор дочитан до конца.
        """
        system_prompt, num_predict, mode_name = self._choose_mode(user_text)
        print(f"[LLM] Режим: {mode_name} (num_predict={num_predict}, num_ctx={OLLAMA_NUM_CTX}, "
              f"потоков: {OLLAMA_THREADS})")

        if reset_history:
            self.history.clear()

        messages = [{"role": "system", "content": system_prompt},
                    *self.history,
                    {"role": "user", "content": user_text}]

        print(f"[LLM] Запрос к {OLLAMA_MODEL}...")
        t0 = time.time()
        t_first = None
        buffer, sentences, truncated = "", [], False
        try:
            for part in chat(
                model=OLLAMA_MODEL,
                messages=messages,
                stream=True,
                options={
                    "temperature": 0.5,
                    "num_predict": num_predict,
                    "num_ctx": OLLAMA_NUM_CTX,
                    "num_thread": OLLAMA_THREADS,
                    "top_k": 40,
                    "top_p": 0.9,
                },
                keep_alive=OLLAMA_KEEP_ALIVE,
            ):
                buffer += part["message"]["content"]
                if part.get("done") and part.get("done_reason") == "length":
                    truncated = True
                while (cut := self._split_point(buffer)) is not None:
                    sentence, buffer = buffer[:cut].strip(), buffer[cut:]
                    if t_first is None:
                        t_first = time.time() - t0
                        print(f"[LLM] Первое предложение через {t_first:.2f} сек")
                    sentences.append(sentence)
                    yield sentence
        except Exception as e:
            print(f"[LLM] Ошибка: {e}")
            if not sentences:
                buffer = "Извините, произошла ошибка при генерации ответа."

        tail = buffer.strip()
        if tail and truncated and sentences and tail[-1] not in ".!?…":
            # Упёрлись в num_predict посреди предложения — недописанный хвост не озвучиваем
            print(f"[LLM] Ответ упёрся в лимит токенов — отброшен хвост: {tail!r}")
            tail = ""
        if not tail and not sentences:
            tail = "Извините, я не смог сформулировать ответ."
        if tail:
            sentences.append(tail)
            yield tail

        answer = " ".join(sentences)
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": answer})
        print(f"[LLM] Ответ за {time.time() - t0:.2f} сек ({len(answer.split())} слов, "
              f"предложений: {len(sentences)}): {answer!r}")

    def ask_llm(self, user_text: str, reset_history: bool = False) -> str:
        """Ответ целиком (для CLI)."""
        return " ".join(self.ask_llm_stream(user_text, reset_history=reset_history))

    # ---------- TTS ----------
    def load_tts(self):
        if self.tts_model is None:
            print(f"[TTS] Загружаю Silero v5 (CPU, потоков: {TTS_THREADS})...")
            torch.set_num_threads(TTS_THREADS)
            t0 = time.time()
            self.tts_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_tts",
                language="ru",
                speaker="v5_ru",
                trust_repo=True,
            )
            print(f"[TTS] Готово за {time.time() - t0:.1f} сек")
        return self.tts_model

    @staticmethod
    def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        """Приводит аудио к нужной частоте дискретизации (полифазный фильтр, 48→16 кГц точно 1:3)."""
        if src_sr == dst_sr:
            return audio
        g = gcd(src_sr, dst_sr)
        return scipy.signal.resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)

    def _tts_piece(self, text: str) -> np.ndarray:
        try:
            audio = self.tts_model.apply_tts(
                text=text,
                speaker=SILERO_SPEAKER,
                sample_rate=SILERO_SR,
                put_accent=True,
                put_yo=True,
            )
        except TypeError:
            # Fallback: если параметры put_accent/put_yo недоступны
            audio = self.tts_model.apply_tts(
                text=text,
                speaker=SILERO_SPEAKER,
                sample_rate=SILERO_SR,
            )
        return audio.numpy() if hasattr(audio, "numpy") else np.array(audio, dtype=np.float32)

    def _tts_16k(self, text: str) -> tuple[np.ndarray, int]:
        """Текст → float32 16 кГц. Возвращает (аудио, число частей)."""
        self.load_tts()

        # 1. Текст для озвучки: числа словами, латиница кириллицей
        spoken = normalize_for_tts(text)
        if spoken != text:
            print(f"[TTS] Текст для озвучки: {spoken!r}")

        # 2. Синтез по частям (Silero не принимает слишком длинный текст)
        parts = split_for_tts(spoken)
        if not parts:
            raise ValueError(f"Нечего озвучивать после нормализации: {text!r}")
        pause = np.zeros(int(SILERO_SR * TTS_PAUSE_SEC), dtype=np.float32)
        pieces = []
        for i, part in enumerate(parts):
            if i:
                pieces.append(pause)
            pieces.append(self._tts_piece(part))
        audio_np = np.concatenate(pieces)

        # 3. Ресемплинг 48→16 кГц
        audio_16k = self._resample(audio_np, SILERO_SR, TARGET_SR)

        # 4. Нормализация громкости (усиление ограничено, чтобы тихие фразы не «шумели»)
        peak = float(np.max(np.abs(audio_16k)))
        if peak > 0:
            audio_16k = audio_16k * min(0.95 / peak, 3.0)
        return audio_16k, len(parts)

    def tts_pcm16(self, text: str) -> bytes:
        """Текст → PCM16 little-endian 16 кГц (для отправки на ESP32 без записи в файл)."""
        t0 = time.time()
        audio_16k, _ = self._tts_16k(text)
        pcm = (np.clip(audio_16k, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        print(f"[TTS] {len(audio_16k) / TARGET_SR:.2f} сек аудио за {time.time() - t0:.2f} сек: {text[:60]!r}")
        return pcm

    def synthesize(self, text: str, out_path: Path) -> Path:
        print(f"[TTS] Синтез ({SILERO_SPEAKER})...")
        t0 = time.time()
        audio_16k, parts = self._tts_16k(text)

        # Сохранение PCM16 16 кГц
        sf.write(str(out_path), audio_16k, TARGET_SR, subtype="PCM_16")

        dur = len(audio_16k) / TARGET_SR
        print(f"[TTS] Сохранено за {time.time() - t0:.2f} сек: {out_path} "
              f"({dur:.2f} сек аудио, {TARGET_SR} Гц, частей: {parts})")
        return out_path

    # ---------- ПОЛНЫЙ ЦИКЛ ----------
    def process_file(self, input_wav: Path, tag: str = "answer",
                     reset_history: bool = False) -> dict:
        print("\n" + "=" * 60)
        print(f"ОБРАБОТКА: {input_wav.name}")
        print("=" * 60)

        t_start = time.time()

        # 1. STT
        user_text = self.transcribe(input_wav)
        if not user_text:
            print("[!] Не удалось распознать речь.")
            return {"user_text": "", "answer": "", "audio": None, "time": 0}

        # 2. LLM (адаптивная длина)
        answer = self.ask_llm(user_text, reset_history=reset_history)

        # 3. TTS
        out_wav = OUTPUT_DIR / f"{tag}.wav"
        self.synthesize(answer, out_wav)

        # 4. Сохраняем текст ответа
        (OUTPUT_DIR / f"{tag}.txt").write_text(answer, encoding="utf-8")

        elapsed = time.time() - t_start
        print("\n" + "=" * 60)
        print(f"[✓] ГОТОВО за {elapsed:.1f} сек")
        print(f"    Вопрос: {user_text}")
        print(f"    Ответ:  {answer}")
        print(f"    Аудио:  {out_wav}")
        print("=" * 60 + "\n")

        return {"user_text": user_text, "answer": answer,
                "audio": out_wav, "time": elapsed}


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Голосовой ассистент: WAV → ответ WAV"
    )
    parser.add_argument("input", type=Path, nargs="?", help="Входной WAV-файл с вопросом")
    parser.add_argument("--tag", default="answer", help="Префикс выходных файлов")
    parser.add_argument("--reset", action="store_true",
                        help="Сбросить историю диалога перед запросом")
    parser.add_argument("--no-preload", action="store_true",
                        help="Не прогревать модели при старте")
    parser.add_argument("--beam", type=int, default=WHISPER_BEAM,
                        help="beam_size для STT: 1=быстро, 5=точно")
    parser.add_argument("--normalize", metavar="TEXT",
                        help="Только показать, как текст будет озвучен (без моделей)")
    args = parser.parse_args()

    if args.normalize:
        print(normalize_for_tts(args.normalize))
        return

    if args.input is None:
        parser.error("нужен входной WAV-файл")
    if not args.input.exists():
        raise FileNotFoundError(f"Файл не найден: {args.input}")

    assistant = VoiceAssistant(
        preload=not args.no_preload,
        beam_size=args.beam,
    )
    assistant.process_file(
        args.input,
        tag=args.tag,
        reset_history=args.reset,
    )


if __name__ == "__main__":
    main()
