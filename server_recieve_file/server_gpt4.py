from flask import Flask
from flask_socketio import SocketIO
import wave
import time
import numpy as np
import pyaudio
from flask_cors import CORS
from engineio.payload import Payload
import threading
import tensorflow as tf
from tensorflow import keras
import os
import pickle
import logging
from scipy.signal import stft, istft

# Setup Logging
logging.basicConfig(level=logging.INFO)

# Model and Audio Configuration
frame_length = 384
frame_step = 160
fft_length = 384

# Flask and SocketIO Setup
app = Flask(__name__)
CORS(app)
Payload.max_decode_packets = 5000
socketio = SocketIO(app, cors_allowed_origins="*")

# Threading Lock for Frames
frames_lock = threading.Lock()

# File Paths
data_path = "../"
out_path = "../"
vocab_path_inp = "G:/FYP/Dataset/char_to_num_vocab_v2.pkl"
vocab_path_out = "G:/FYP/Dataset/char_to_num_vocab_v2.pkl"
model_checkpoint_path = r"G:/FYP/models/model_checkpoint_v2.h5"

# Load Vocabulary
loaded_vocab = None
if os.path.exists(vocab_path_inp):
    with open(vocab_path_inp, "rb") as f:
        loaded_vocab = pickle.load(f)

# Character-to-Number and Number-to-Character Mappings
char_to_num = keras.layers.StringLookup(vocabulary=loaded_vocab, oov_token="")
num_to_char = keras.layers.StringLookup(vocabulary=loaded_vocab, oov_token="", invert=True)

# CTC Loss Function
def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

# Utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text

# Load Model
if os.path.exists(model_checkpoint_path):
    custom_objects = {"CTCLoss": CTCLoss}
    with keras.utils.custom_object_scope(custom_objects):
        model = keras.models.load_model(model_checkpoint_path)

# Function to Encode a Single Sample
def encode_single_sample(wav_file, label=None):
    # Process the Audio
    file = tf.io.read_file(wav_file)
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cast(audio, tf.float32)
    spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)

    # Process the Label
    if label is not None:
        label = tf.strings.lower(label)
        label = tf.strings.unicode_split(label, input_encoding="UTF-8")
        label = char_to_num(label)
        return spectrogram, label
    return spectrogram



# Function to Transcribe Frames
def transcribe_frames(frames):

    with frames_lock:
        save_wav(frames, 'audio1.wav')

        logging.info("Audio saved, starting transcription.")
        
        spectrogram = encode_single_sample("audio.wav")
        spectograms = tf.expand_dims(spectrogram, axis=0)
        batch_predictions1 = model.predict(spectograms)
        batch_predictions = decode_batch_predictions(batch_predictions1)
        logging.info(f"Transcription: {batch_predictions}")
        socketio.emit('audio_dt', f'{batch_predictions}')
  
# Audio Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK = 4096 * 10
audio1 = pyaudio.PyAudio()
RATE = 15000
threshold = 199305.0
ms = 1200
last_speech_time = time.time()
start = False

def check_thresh(a):
    audio_buffer = np.frombuffer(a, dtype=np.int16)
    b = ((np.abs(np.fft.rfft(audio_buffer))) * 16000 / (len(a)//2)).mean()
    return b > threshold 

def save_wav(frames, filename, chunk=CHUNK, sample_format=FORMAT, channels=CHANNELS, fs=RATE):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    try:
        wf.setsampwidth(audio1.get_sample_size(sample_format))
    except Exception as e:
        print(str(e))
        pass
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

# SocketIO Event Handler for Audio Data
@socketio.on('audio_data', namespace='/')
def handle_audio_data(audio_data):
    global frames_in, silence_duration, last_speech_time, start
    if audio_data:
        print("Getting Audio Data")
    check = check_thresh(audio_data)
    if check:
        last_speech_time = time.time()
        frames_in.append(audio_data)
        silence_duration = 0
        start = True
        logging.info(f'User is speaking: {len(frames_in)} frames')
    else:
        silence_duration = time.time() - last_speech_time
        if start:
            frames_in.append(audio_data)

    if silence_duration * 1000 >= ms and start:
        start = False
        logging.info(f"Silence for {silence_duration * 1000}ms, processing...")
        try:
            transcribe_frames(frames_in)
            frames_in = []
        except:
            frames_in = []
    return "Audio Data Received"

# Global Variables
frames_in = []
silence_duration = 0

# Starting Flask and SocketIO
if __name__ == '__main__':
    logging.info("Starting server...")
    socketio.run(app, port=5031)
