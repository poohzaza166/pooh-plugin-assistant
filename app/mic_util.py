import asyncio
import io
import json
import queue
import threading
import wave
from collections import deque

import requests
import sounddevice as sd
from bus import MessageBus
from vosk import KaldiRecognizer, Model


class WakeWordWhisper:
    def __init__(
        self,
        bus: MessageBus,
        wake_words=["scarlet", "hey scarlet",
                    "scarlett", "hey scarlett", "[unk]"],
        whisper_server_url="http://127.0.0.1:8080/inference",
        sample_rate=16000,
        channels=1,
        block_size=2000,
        pre_roll=0.1,
        silence_duration=1,
        silence_threshold=1000,
        vosk_model_path="vosk-model-small-en-us-0.15",
    ):
        # Config
        self.SAMPLE_RATE = sample_rate
        self.CHANNELS = channels
        self.BLOCK_SIZE = block_size
        self.WAKE_WORDS = [w.lower() for w in (wake_words)]
        self.WHISPER_SERVER_URL = whisper_server_url
        self.PRE_ROLL = pre_roll
        self.SILENCE_DURATION = silence_duration
        self.SILENCE_THRESHOLD = silence_threshold

        # Vosk setup
        self.model = Model(vosk_model_path)
        self.rec = KaldiRecognizer(self.model, self.SAMPLE_RATE)
        self.rec.SetWords(False)

        # Buffers
        self.ring = deque(maxlen=int(self.PRE_ROLL * self.SAMPLE_RATE * 2))
        self.audio_queue = queue.Queue()
        self.record_buffer = []
        self.triggered = False
        self.silence_counter = 0.0

        self.bus = bus

        # Threads
        self.worker_thread = threading.Thread(
            target=self.audio_worker, daemon=True)
        self.stream_thread = threading.Thread(
            target=self.start_audio_stream, daemon=True)

    # ===== Audio processing =====
    def calculate_rms(self, pcm_bytes):
        samples = memoryview(pcm_bytes).cast("h")
        if not samples:
            return 0
        return (sum(s*s for s in samples) / len(samples)) ** 0.5

    def is_speech(self, pcm_bytes):
        return self.calculate_rms(pcm_bytes) > self.SILENCE_THRESHOLD

    def send_to_whisper(self, audio_bytes):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(audio_bytes)
        buf.seek(0)
        files = {"file": ("speech.wav", buf, "audio/wav")}
        data = {"temperature": "0.0", "response_format": "json"}
        resp = requests.post(self.WHISPER_SERVER_URL, files=files, data=data)
        resp.raise_for_status()
        print("Whisper transcription:", resp.json().get("text", ""))
        asyncio.run(self.publish_utterance(resp.json().get("text", "")))

    # ===== Worker thread =====
    def audio_worker(self):
        while True:
            pcm = self.audio_queue.get()
            self.ring.append(pcm)

            # Wake word detection
            if not self.triggered and self.rec.AcceptWaveform(pcm):
                text = json.loads(self.rec.Result()).get("text", "")
                if any(wake_word in text for wake_word in self.WAKE_WORDS):
                    print(f"Wake word '{self.WAKE_WORDS}' detected!")
                    self.triggered = True
                    self.record_buffer = list(self.ring)
                    self.silence_counter = 0.0
                    continue

            # Recording after wake word
            if self.triggered:
                self.record_buffer.append(pcm)
                if self.is_speech(pcm):
                    self.silence_counter = 0.0
                else:
                    self.silence_counter += len(pcm) / 2 / self.SAMPLE_RATE

                if self.silence_counter >= self.SILENCE_DURATION:
                    self.send_to_whisper(b"".join(self.record_buffer))
                    self.triggered = False
                    self.record_buffer.clear()
                    self.silence_counter = 0.0

    async def publish_utterance(self, text: str):
        await self.bus.publish_async("intent.query", {
            "utterance": text,
            "context": None
        })

    # ===== Audio callback =====
    def callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_queue.put(bytes(indata))

    # ===== Audio stream =====
    def start_audio_stream(self):
        with sd.RawInputStream(
            samplerate=self.SAMPLE_RATE,
            blocksize=self.BLOCK_SIZE,
            dtype="int16",
            channels=self.CHANNELS,
            callback=self.callback,
        ):
            print(f"🎙 Listening for wake word '{self.WAKE_WORDS}'...")
            while True:
                sd.sleep(1000)

    # ===== Start threads =====
    def start(self):
        self.worker_thread.start()
        self.stream_thread.start()


if __name__ == "__main__":
    ww = WakeWordWhisper(
        wake_words=["computer", "hello assistant"]
    )
    ww.start()
    import asyncio

    async def main_loop():
        while True:
            print("hello from asyncio!")
            await asyncio.sleep(1)

    asyncio.run(main_loop())
