#!/usr/bin/env python3
"""
Voice Chat - Full voice conversation loop with Claude
Listens → Transcribes (Whisper) → Sends to API → Types + Speaks response

Requirements:
    pip install sounddevice numpy whisper openai-whisper

Usage:
    python voice_chat.py
"""

import subprocess
import sys
import threading
import time
import queue
import tempfile
import os

# Check dependencies
def check_deps():
    missing = []
    try:
        import sounddevice
    except ImportError:
        missing.append("sounddevice")
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    try:
        import whisper
    except ImportError:
        missing.append("openai-whisper")

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)

check_deps()

import sounddevice as sd
import numpy as np
import whisper

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 1.5  # seconds of silence before processing
MAX_RECORDING_TIME = 30  # max seconds per recording

# ANSI colors
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

print(f"{Colors.CYAN}Loading Whisper model (this may take a moment)...{Colors.END}")
model = whisper.load_model("base")  # Options: tiny, base, small, medium, large
print(f"{Colors.GREEN}Whisper loaded!{Colors.END}")


def record_until_silence():
    """Record audio until user stops speaking"""
    print(f"\n{Colors.YELLOW}🎤 Listening... (speak now){Colors.END}")

    audio_buffer = []
    silence_samples = 0
    silence_threshold_samples = int(SILENCE_DURATION * SAMPLE_RATE)
    max_samples = MAX_RECORDING_TIME * SAMPLE_RATE
    total_samples = 0

    def callback(indata, frames, time_info, status):
        nonlocal silence_samples, total_samples
        audio_buffer.append(indata.copy())

        # Check for silence
        volume = np.abs(indata).mean()
        if volume < SILENCE_THRESHOLD:
            silence_samples += frames
        else:
            silence_samples = 0
        total_samples += frames

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback):
        while silence_samples < silence_threshold_samples and total_samples < max_samples:
            time.sleep(0.1)

    if len(audio_buffer) == 0:
        return None

    audio = np.concatenate(audio_buffer)
    print(f"{Colors.GREEN}✓ Recording complete ({len(audio)/SAMPLE_RATE:.1f}s){Colors.END}")
    return audio


def transcribe(audio):
    """Transcribe audio using Whisper"""
    print(f"{Colors.CYAN}📝 Transcribing...{Colors.END}")

    # Normalize audio
    audio = audio.flatten().astype(np.float32)

    # Transcribe
    result = model.transcribe(audio, fp16=False)
    text = result["text"].strip()

    print(f"{Colors.BOLD}You said:{Colors.END} {text}")
    return text


def speak_and_type(text):
    """Speak text while typing it character by character"""
    # Start speaking in background using macOS default voice
    speak_thread = threading.Thread(target=lambda: subprocess.run(
        ["say", text],
        capture_output=True
    ))
    speak_thread.start()

    # Type out response
    print(f"\n{Colors.GREEN}🤖 Claude:{Colors.END} ", end="", flush=True)

    words = text.split()
    for i, word in enumerate(words):
        print(word, end=" ", flush=True)
        # Approximate speech timing
        time.sleep(len(word) * 0.05 + 0.1)

    print()  # newline at end
    speak_thread.join()


def send_to_chat(message):
    """Send message to local backend chat API"""
    import json
    import urllib.request

    # Use local backend - it already has Claude API access configured
    url = "http://localhost:8000/api/chat/simple"
    headers = {
        "Content-Type": "application/json",
    }

    data = json.dumps({
        "message": message,
        "system": "You are a helpful voice assistant. Keep responses concise and conversational - they will be spoken aloud. Aim for 1-3 sentences unless more detail is needed."
    }).encode()

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())
            return result.get("response", result.get("message", "No response"))
    except urllib.error.URLError as e:
        return f"Couldn't connect to backend. Is it running? (python backend/main.py)"
    except Exception as e:
        return f"Sorry, I had trouble connecting: {str(e)}"


def main():
    print(f"""
{Colors.BOLD}╔══════════════════════════════════════╗
║       Voice Chat with Claude         ║
╚══════════════════════════════════════╝{Colors.END}

{Colors.CYAN}Commands:{Colors.END}
  • Just speak naturally after the "Listening" prompt
  • Say "goodbye" or "exit" to quit
  • Press Ctrl+C to force quit

{Colors.GREEN}Ready to chat!{Colors.END}
""")

    while True:
        try:
            # Record audio
            audio = record_until_silence()

            if audio is None or len(audio) < SAMPLE_RATE * 0.5:
                print(f"{Colors.RED}(Too short, try again){Colors.END}")
                continue

            # Transcribe
            text = transcribe(audio)

            if not text or len(text) < 2:
                continue

            # Check for exit
            if any(word in text.lower() for word in ["goodbye", "exit", "quit", "bye"]):
                speak_and_type("Goodbye! Talk to you later.")
                break

            # Get response from Claude
            print(f"{Colors.CYAN}🧠 Thinking...{Colors.END}")
            response = send_to_chat(text)

            # Speak and type response
            speak_and_type(response)

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Interrupted. Goodbye!{Colors.END}")
            break
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.END}")
            continue


if __name__ == "__main__":
    main()
