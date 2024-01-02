from flask import Flask, session
from flask_socketio import SocketIO
import wave
import time
import numpy as np
import pyaudio
from flask_cors import CORS
from engineio.payload import Payload
import threading
import tensorflow as tf
import os
from tensorflow import keras
import pickle
import logging
import sys
print(os.getcwd())
from preprocess import Preporcess_Data  # Import your preprocessing class
from load_dataset import Load_Data
# ...

# Instantiate Preporcess_Data with correct paths and settings

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
load_data = Load_Data()
char_to_num= load_data.char_to_num
preprocessor = Preporcess_Data(wavs_path="G:/FYP/Dataset/cv-corpus-14.0-2023-06-23/ur/limited_wav_files/", char_to_num=char_to_num, frame_length=256, frame_step=160, fft_length=384, predict=True)
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
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss



# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    print(results,"*"*50)
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text

# Load Model
if os.path.exists(model_checkpoint_path):
    custom_objects = {"CTCLoss": CTCLoss}
    with keras.utils.custom_object_scope(custom_objects):
        model = keras.models.load_model(model_checkpoint_path)

def encode_single_sample(wav_file, label=None):
        ###########################################
        ##  Process the Audio
        ##########################################
        # 1. Read wav

        file = tf.io.read_file( wav_file)
        # 2. Decode the wav file
        audio, _ = tf.audio.decode_wav(file)
        audio = tf.squeeze(audio, axis=-1)
        # 3. Change type to float
        audio = tf.cast(audio, tf.float32)
        # 4. Get the spectrogram
        spectrogram = tf.signal.stft(
            audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
        )
        # 5. We only need the magnitude, which can be derived by applying tf.abs
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.math.pow(spectrogram, 0.5)
        # 6. normalisation
        means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
        stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
        spectrogram = (spectrogram - means) / (stddevs + 1e-10)
        ###########################################
        ##  Process the label
        ##########################################
        # 7. Convert label to Lower case
        if label is not None:
            label = tf.strings.lower(label)
            # 8. Split the label
            label = tf.strings.unicode_split(label, input_encoding="UTF-8")
            # 9. Map the characters in label to numbers
            label = char_to_num(label)
            # 10. Return a dict as our model is expecting two inputs
            return spectrogram, label
        return spectrogram

# Function to Transcribe Frames
def transcribe_frames():
    with frames_lock:
        # Saving frames to a WAV file
        # save_wav(frames, 'audio.wav')
        logging.info("Audio saved, starting transcription.")
        
        # Load and Transcribe using the Preporcess_Data class
        spectrogram = preprocessor.encode_single_sample('common_voice_ur_26562732')
        spectograms = tf.expand_dims(spectrogram, axis=0)
        batch_predictions1 = model.predict(spectograms)
        batch_predictions = decode_batch_predictions(batch_predictions1)
        print(f"Transcription: {batch_predictions}")
        logging.info(f"Transcription: {batch_predictions}")
        socketio.emit('audio_dt', f'{batch_predictions}')

# Audio Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK = 4096 * 10
audio1 = pyaudio.PyAudio()
RATE = 15000
threshold = 70000#1499305.0
ms = 500
last_speech_time = time.time()
start = False
def check_thresh(a):
    audio_buffer = np.frombuffer(a, dtype=np.int16)
    b = ((np.abs(np.fft.rfft(audio_buffer))) * 16000 / (len(a)//2)).mean()
    return b > threshold 
def save_wav(frames, filename, chunk = CHUNK  ,sample_format =FORMAT, channels = CHANNELS, fs = RATE):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    try:
        wf.setsampwidth(audio1.get_sample_size(sample_format))
    except Exception as e:
        print(str(e))
        pass
    wf.setframerate(fs)
    # time.sleep(0.1)
    wf.writeframes(b''.join(frames))
#     frames.clear()
    wf.close()
# Define SocketIO event handler for audio data
# SocketIO Event Handler for Audio Data
# @socketio.on('audio_data', namespace='/')
# def handle_audio_data(audio_data):
#     global frames_in, silence_duration, last_speech_time, start
#     if audio_data:
#         print("Getting Audio Data")
#     check = check_thresh(audio_data)
#     if check:
#         last_speech_time = time.time()
#         frames_in.append(audio_data)
#         silence_duration = 0
#         start = True
#         logging.info(f'User is speaking: {len(frames_in)} frames')
#     else:
#         silence_duration = time.time() - last_speech_time
#         if start:
#             frames_in.append(audio_data)

#     if silence_duration * 1000 >= ms and start:
#         if len(frames_in) >= 30:
#             logging.info("Silence detected, processing audio.")
#             transcription_thread = threading.Thread(target=transcribe_frames, args=(frames_in.copy(),))
#             transcription_thread.start()
#             frames_in.clear()
#         start = False

# # Main Function to Run Server
# if __name__ == '__main__':
#     frames_in = []  # List to store audio frames
#     logging.info("Starting server...")
#     socketio.run(app, host='0.0.0.0', port=5031)
p = transcribe_frames()