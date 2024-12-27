from pyht import Client
from dotenv import load_dotenv
from pyht.client import TTSOptions
from scipy.io.wavfile import write
import os
import numpy as np
load_dotenv()


client = Client(
    user_id="LxQ8AxVqfJaZTOglsWlsX66DHVw2",
    api_key="2170240dcb6a4cc59ce158bd07f7ba48"
)


audio_chunks = []

options = TTSOptions(voice="s3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/female-cs/manifest.json")
for chunk in client.tts("Can you tell me your account email or, ah your phone number?", options):
    
    audio_chunks.append(chunk)

with open('myfile.wav', mode='bx') as f:
    f.write(audio_chunks[0])
