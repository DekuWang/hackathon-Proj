import pyaudio
import wave
import numpy as np
import time

# === Config ===
THRESHOLD = 80            # Adjust this for your microphone sensitivity
SILENCE_LIMIT = 2         # Seconds of silence before stopping
CHUNK = 1024              # Buffer size
FORMAT = pyaudio.paInt16  # 16-bit PCM
CHANNELS = 1              # Mono
RATE = 16000              # 16kHz
OUTPUT_FILENAME = "output.wav"

def rms(data):
    """Root mean square of audio buffer"""
    audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    return np.sqrt(np.mean(audio_data**2))

def record(output = OUTPUT_FILENAME):
    """Listen for voice and record audio until silence is detected."""
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Listening for voice...")

    frames = []
    recording = False
    silence_start = None

    try:
        while True:
            data = stream.read(CHUNK)
            volume = rms(data)

            if volume > THRESHOLD:
                if not recording:
                    print("[INFO] Voice detected. Start recording...")
                    recording = True
                frames.append(data)
                silence_start = None  # reset silence timer

            elif recording:
                frames.append(data)
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_LIMIT:
                    print("[INFO] Silence detected. Stop recording.")
                    break

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

        if frames:
            print(f"[INFO] Saving to {output}")
            wf = wave.open(output, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
        else:
            print("[INFO] No audio recorded.")

if __name__ == "__main__":
    record()
