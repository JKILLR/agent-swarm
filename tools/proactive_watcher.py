#!/usr/bin/env python3
"""
Proactive Screen Watcher
Takes periodic screenshots and sends them to the COO for analysis.
The COO will speak observations/suggestions via the say command.
"""

import subprocess
import base64
import requests
import time
import os
from datetime import datetime
from pathlib import Path

# Configuration
CHECK_INTERVAL = 300  # 5 minutes in seconds
SCREENSHOT_PATH = "/tmp/proactive_screenshot.png"
BACKEND_URL = "http://localhost:8000"

def take_screenshot():
    """Capture the current screen."""
    try:
        subprocess.run([
            "screencapture", "-x", SCREENSHOT_PATH
        ], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Screenshot failed: {e}")
        return False

def get_screenshot_base64():
    """Read screenshot and convert to base64."""
    with open(SCREENSHOT_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def speak(text):
    """Speak text using macOS say command."""
    # Use Samantha voice for a friendly tone
    subprocess.run(["say", "-v", "Samantha", text])

def analyze_screen():
    """Send screenshot to COO for analysis."""
    if not take_screenshot():
        return None

    screenshot_b64 = get_screenshot_base64()

    prompt = f"""You are doing a proactive check-in on J. You just captured their screen.

Current time: {datetime.now().strftime("%I:%M %p")}

Analyze what they're working on and provide a brief, natural spoken response. You can:
- Offer help if they seem stuck on something
- Make an observation about their work
- Gently nudge if they seem distracted
- Just say hi and ask how it's going
- Point out something interesting you notice

Keep it SHORT (1-2 sentences max) and conversational. You'll speak this via text-to-speech.

Don't be annoying or preachy. Be a helpful friend checking in, not a nagging assistant.

If they seem deep in focused work, maybe just a quick "Looking good, carry on!" is enough.
If they're on social media for a while, a gentle "Taking a break?" is fine.

Respond with ONLY what you want to say out loud. No explanations or metadata."""

    try:
        response = requests.post(
            f"{BACKEND_URL}/api/chat",
            json={
                "message": prompt,
                "images": [f"data:image/png;base64,{screenshot_b64}"]
            },
            timeout=60
        )

        if response.status_code == 200:
            data = response.json()
            return data.get("response", "").strip()
        else:
            print(f"API error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None

def run_watcher():
    """Main loop for the proactive watcher."""
    print(f"🔭 Proactive Watcher started at {datetime.now().strftime('%I:%M %p')}")
    print(f"📸 Checking in every {CHECK_INTERVAL // 60} minutes")
    print("Press Ctrl+C to stop\n")

    # Initial greeting
    speak("Hey J, proactive watcher is now running. I'll check in every 5 minutes.")

    while True:
        try:
            time.sleep(CHECK_INTERVAL)

            print(f"\n[{datetime.now().strftime('%I:%M %p')}] Checking in...")

            response = analyze_screen()

            if response:
                print(f"Speaking: {response}")
                speak(response)
            else:
                print("No response generated")

        except KeyboardInterrupt:
            print("\n\n👋 Watcher stopped. See you later!")
            speak("Proactive watcher stopped. Talk to you later!")
            break
        except Exception as e:
            print(f"Error during check-in: {e}")
            time.sleep(10)  # Brief pause before retrying

if __name__ == "__main__":
    run_watcher()
