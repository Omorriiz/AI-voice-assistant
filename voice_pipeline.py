"""
Голосовой ассистент: STT (Faster-Whisper) -> LLM (Ollama) -> TTS (Silero v5).
"""

import os
import time
import argparse
from pathlib import Path

# --- отключаем XET для HF, чтобы не тянуло лишнее ---
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
import soundfile as sf
from faster_whisper import WhisperModel
from ollama import chat


# ============================================================
# CONFIG
# ============================================================

WHISPER_MODEL_DIR = r"C:/Users/yumit/Desktop/Degree/server/model_faster_Whisper"
WHISPER_LANG      = "ru"

OLLAMA_MODEL      = "OxW/Vikhr-Nemo-12B-Instruct-R-21-09-24:q4_K_M"

SILERO_SPEAKER    = "xenia"
SILERO_SR         = 48000

SYSTEM_PROMPT = (
    "Ты голосовой ассистент. Отвечай кратко, ясно и по делу. "
    "Не используй списки, маркдаун и эмодзи — ответ будет озвучен вслух.Все числа пиши словами."
)

OUTPUT_DIR = Path(r"C:\Users\yumit\Desktop\Degree\server\output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# КЛАСС-ОБЁРТКА
# ============================================================

class VoiceAssistant:
    def __init__(self):
        self.whisper = None
        self.tts_model = None
        self.history = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

    # ---------- STT ----------
    def load_whisper(self):
        if self.whisper is None:
            print("[STT] Загружаю Faster-Whisper...")
            self.whisper = WhisperModel(
                WHISPER_MODEL_DIR,
                device="cpu",
                compute_type="int8",
            )
            print("[STT] Готово.")
        return self.whisper

    def transcribe(self, wav_path: Path) -> str:
        self.load_whisper()
        print(f"[STT] Распознаю: {wav_path}")
        segments, _info = self.whisper.transcribe(
            str(wav_path),
            language=WHISPER_LANG,
            task="transcribe",
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(s.text.strip() for s in segments if s.text.strip()).strip()
        print(f"[STT] Распознано: {text!r}")
        return text

    # ---------- LLM ----------
    def ask_llm(self, user_text: str, reset_history: bool = False) -> str:
        if reset_history:
            self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

        self.history.append({"role": "user", "content": user_text})

        print(f"[LLM] Запрос к {OLLAMA_MODEL}...")
        response = chat(
            model=OLLAMA_MODEL,
            messages=self.history,
            options={
                "temperature": 0.5,
                "num_predict": 200,
            },
        )
        answer = response["message"]["content"].strip()
        self.history.append({"role": "assistant", "content": answer})
        print(f"[LLM] Ответ: {answer!r}")
        return answer

    # ---------- TTS ----------
    def load_tts(self):
        if self.tts_model is None:
            print("[TTS] Загружаю Silero v5...")
            self.tts_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_tts",
                language="ru",
                speaker="v5_ru",
                trust_repo=True,
            )
            print("[TTS] Готово.")
        return self.tts_model

    def synthesize(self, text: str, out_path: Path) -> Path:
        self.load_tts()
        print(f"[TTS] Синтез ({SILERO_SPEAKER})...")
        audio = self.tts_model.apply_tts(
            text=text,
            speaker=SILERO_SPEAKER,
            sample_rate=SILERO_SR,
            put_accent=True,
            put_yo=True,
        )
        sf.write(str(out_path), audio.numpy(), SILERO_SR)
        dur = len(audio) / SILERO_SR
        print(f"[TTS] Сохранено: {out_path} ({dur:.2f} сек)")
        return out_path

    # ---------- ПОЛНЫЙ ЦИКЛ ----------
    def process_file(self, input_wav: Path, tag: str = "answer",
                     reset_history: bool = False) -> dict:
        t0 = time.time()

        user_text = self.transcribe(input_wav)
        if not user_text:
            print("[!] Не удалось распознать речь.")
            return {"user_text": "", "answer": "", "audio": None}

        answer = self.ask_llm(user_text, reset_history=reset_history)

        out_wav = OUTPUT_DIR / f"{tag}.wav"
        self.synthesize(answer, out_wav)

        # сохраняем текст ответа рядом
        (OUTPUT_DIR / f"{tag}.txt").write_text(answer, encoding="utf-8")

        print(f"[✓] Готово за {time.time() - t0:.1f} сек")
        return {"user_text": user_text, "answer": answer, "audio": out_wav}


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Голосовой ассистент: WAV -> ответ WAV")
    parser.add_argument("input", type=Path, help="Входной WAV-файл с вопросом")
    parser.add_argument("--tag", default="answer", help="Префикс выходных файлов")
    parser.add_argument("--reset", action="store_true",
                        help="Сбросить историю диалога перед запросом")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Файл не найден: {args.input}")

    assistant = VoiceAssistant()
    assistant.process_file(args.input, tag=args.tag, reset_history=args.reset)


if __name__ == "__main__":
    main()