import pyaudio
# Used by main.py and face_catcher.py
USERNAME = "Bo Zhang"
MODE = "recognition"  # recognition mode / enrollment mode / update mode
IS_ENROLLMENT = False  # Used by face_catcher.py
IS_UPDATE = False  # Used by face_catcher.py
UNKNOWN_COUNTER_THRESHOLD = 50  # Used by face_catcher.py

# Used by face_catcher.py
LBPH_MODEL = r"pretrained_model\haarcascade_frontalface_alt.xml"

# Used by LLM.py
API_KEY = """
""".strip()

# Used by recoder.py
THRESHOLD = 80            # Adjust this for your microphone sensitivity
SILENCE_LIMIT = 2         # Seconds of silence before stopping
CHUNK = 1024              # Buffer size
FORMAT = pyaudio.paInt16  # 16-bit PCM
CHANNELS = 1              # Mono
RATE = 16000              # 16kHz
OUTPUT_FILENAME = "output.wav"
