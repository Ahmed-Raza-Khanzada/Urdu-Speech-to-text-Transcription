# Python code on the server

import asyncio
import websockets
import numpy as np
from flask import Flask, session
from flask_socketio import SocketIO, emit
from engineio.payload import Payload
import pyaudio
import torch
import time
import numpy as np
import whisper
import wave

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Constants for audio settings
SAMPLE_RATE = 44100
CHANNELS = 1
CHUNK = 4096*10

FORMAT = pyaudio.paFloat32
RATE = 24000
audio = pyaudio.PyAudio()
frames_in = []

print("Loading model...")
model = whisper.load_model("base")
print("Whisper Model Loaded")
def save_wav(frames, filename, chunk = CHUNK  ,sample_format =FORMAT, channels = CHANNELS, fs = RATE):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    try:
        wf.setsampwidth(audio.get_sample_size(sample_format))
    except Exception as e:
        print(str(e))
        pass
    wf.setframerate(fs)
    time.sleep(0.1)
    wf.writeframes(b''.join(frames))
#     frames.clear()
    wf.close()
async def receive_audio(websocket, path):
    global frames_in

    # Accept the WebSocket connection
    await websocket.accept()

    try:
        while True:
            # Receive audio data from the browser
            audio_data = await websocket.recv()

            # Accumulate the received frames
            frames_in.append(audio_data)

            # Process frames in chunks
            buffer_size = 20
            if len(frames_in) >= buffer_size:
                audio_data_np = np.frombuffer(bytes(frames_in[:buffer_size]), np.int16).flatten().astype("float32") / 32768.0

                # Reset frames buffer
                frames_in = frames_in[buffer_size:]
                
                time.sleep(0.2)#0.2
                
                # Process the audio data as needed
                print('Received audio data:', audio_data_np)

    except asyncio.CancelledError:
        pass

start_server = websockets.serve(receive_audio, 'localhost', 8765)

# Define SocketIO event handler for audio data
@socketio.on('audio_data')
def handle_audio_data(audio_data):
    global frames_in
    frames_in.append(audio_data)

    buffer_size = 20
    if len(frames_in) >= buffer_size:
        audio_data_np = np.frombuffer(bytes(frames_in[:buffer_size]), np.int16).flatten().astype("float32") / 32768.0
      
        # Reset frames buffer
        frames_in = frames_in[buffer_size:]
        audio_tensor = torch.from_numpy(audio_data_np)
        save_wav(frames_in, 'audio.wav')
        frames_in.clear()
        print("======================================",audio_tensor,"============================================")
        transcription = model.transcribe(audio_tensor)
        print("****************************************************************************************************"*10)
        print("Transcription is: ",transcription["text"])
        print("***************************************************************")
        socketio.emit('audio_dt', transcription["text"])
        # Process the audio data as needed
        print('Received audio data:', audio_data_np)

# Run Flask-SocketIO server
if __name__ == '__main__':
    print("Starting server...")
    Payload.max_decode_packets = 5000
    socketio.run(app, host='0.0.0.0', port=5031)
