import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from IPython import display
from jiwer import wer
import os
from keras.callbacks import ModelCheckpoint
from pydub import AudioSegment
import pickle


data_path='G:/FYP/'#if path not  changed then /content/drive/MyDrive/Urdu_Speech_wavs/
wavs_path = data_path + "Dataset/cv-corpus-14.0-2023-06-23/ur/limited_wav_files/"
metadata_path = data_path + "Dataset/cv-corpus-14.0-2023-06-23/ur/final_main_dataset.tsv"

# chars_vocab = {'’', 'ى', 'ُ', '”', 'ﷲ', 'ﷺ', 'ض', 'ؤ', '؟', 'ظ', 'ن', 'ﻧ', 'ﺗ', 'و', 'ؓ', 'ه', '`', 'ً', 'ﯾ', 'د', 'ؔ', 'ْ', 'ٰ', 'ﺲ', 'ل', 'ت', '"', 'ش', 'ی', ':', 'ک', "'", 'ء', 'م', 'ٔ', 'ہ', 'ے', 'ژ', 'ۂ', 'ِ', 'ح', 'گ', 'ﺩ', 'چ', 'ص', 'ڈ', 'ﭨ', 'ك', '“', 'ٓ', 'ٗ', ',', 'ي', 'پ', 'ڑ', 'ث', 'َ', 'ف', 'ﮯ', 'ع', 'ب', 'آ', 'ر', '۔', 'ا', 'ھ', 'س', 'ئ', 'ذ', '.', 'أ', '!', 'ز', 'ط', 'خ', 'ﮭ', '،', 'ٹ', 'ۃ', '-', 'ج', 'ﺘ', 'ق', '…', ' ', 'غ', 'ؑ', 'ۓ', '؛', 'ﻮ', '‘', 'ں', 'ّ'}
# characters = list(chars_vocab)
# allow = ["?", " ", "!", "'"]
# alphas = "abcdefghijklmnopqrstuvwxyz"
# characters = [i for i in "".join(characters) if ((i.isalpha()) or (i in allow)) and (i not in alphas)]

# # Creating the character to integer mapping
# char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
# # Saving the vocabulary using pickle
# with open(vocab_path_out, "wb") as f:
#     pickle.dump(char_to_num.get_vocabulary(), f)

# Loading the vocabulary back from pickle
vocab_path_inp = "vocab_path.pkl"
loaded_vocab = None
with open(vocab_path_inp, "rb") as f:
    loaded_vocab = pickle.load(f)

# Creating the integer to character mapping using the loaded vocabulary
char_to_num = keras.layers.StringLookup(vocabulary=loaded_vocab, oov_token="")
num_to_char = keras.layers.StringLookup(vocabulary=loaded_vocab, oov_token="", invert=True)

print(
    f"The loaded vocabulary is: {num_to_char.get_vocabulary()} "
    f"(size ={num_to_char.vocabulary_size()})"
)
print(
    f"The loaded vocabulary is: {char_to_num.get_vocabulary()} "
    f"(size ={char_to_num.vocabulary_size()})"
)

def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss
# Define file paths for model checkpoints
model_checkpoint_path = "model_checkpoint.h5"



# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text



# An integer scalar Tensor. The window length in samples.
frame_length = 256#600#256
# An integer scalar Tensor. The number of samples to step.
frame_step = 160#307#160
# An integer scalar Tensor. The size of the FFT to apply.
# If not provided, uses the smallest power of 2 enclosing frame_length.
fft_length = 384#650#384


def encode_single_sample(wav_file):
    ###########################################
    ##  Process the Audio
    ##########################################
    # 1. Read wav

    file = tf.io.read_file(wavs_path + wav_file+".wav")
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
    # # 7. Convert label to Lower case
    # label = tf.strings.lower(label)
    # # 8. Split the label
    # label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    # # 9. Map the characters in label to numbers
    # label = char_to_num(label)
    # 10. Return a dict as our model is expecting two inputs
    return spectrogram



if os.path.exists(model_checkpoint_path):
    # model = keras.models.load_model(model_checkpoint_path)
    custom_objects = {"CTCLoss": CTCLoss}
    with keras.utils.custom_object_scope(custom_objects):
        model = keras.models.load_model(model_checkpoint_path)

# a = list(os.listdir(wavs_path))[:5]

# # Define the validation dataset
# validation_dataset = tf.data.Dataset.from_tensor_slices(
#    a
# )

# validation_dataset = (
#     validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
#     .padded_batch(len(a))
#     .prefetch(buffer_size=tf.data.AUTOTUNE)
# )


# for batch in validation_dataset:
#     X = batch
#     print(X.shape,"@@@@@@@@@@@@@@@@@@@@")
#     pred = model.predict(X)

#     print(pred[1][1],pred[1].shape,pred[1][1].shape)
#     input_len = np.ones(pred.shape[0]) * pred.shape[1]
#     # Use greedy search. For complex tasks, you can use beam search
#     results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
#     print("*********************************")
#     print(results[0],results.shape)
#     # Iterate over the results and get back the text
#     output_text = []
#     for result in results:
#         result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
#         output_text.append(result)
#     print("----------------------------------------")
#     print(output_text,len(output_text))
#     break



def take_input():
    int1 = input("Enter audio file name:")
    return int1
a = True
while a:
    input1 = take_input()
    if input1=="":
        a = False
        break
    # validation_dataset = tf.data.Dataset.from_tensor_slices(
    # input1)
    wav_file = input1
    spectograms = encode_single_sample(wav_file)

    # Let's check results on more validation samples
    predictions = []
    #reshape spectograms (1,spectograms.shape[0],spectograms.shape[1])
    spectograms=tf.expand_dims(spectograms, axis=0)
    print(spectograms,"@@@@@@@@@@@@@")
    # spectograms = tf.expand_dims(spectograms, axis=0)
    batch_predictions1 = model.predict(spectograms)
    print(batch_predictions1[:2])
    batch_predictions = decode_batch_predictions(batch_predictions1)
    print(batch_predictions,len(batch_predictions))
    predictions.extend(batch_predictions)

    print(predictions,"****************************")




