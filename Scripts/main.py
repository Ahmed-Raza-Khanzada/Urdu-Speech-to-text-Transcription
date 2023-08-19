import os
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from load_dataset import Load_Data
from preprocess import Preporcess_Data
from model import build_model
from utils import CallbackEval,CTCLoss
from jiwer import wer

# Define file paths for model checkpoints
model_checkpoint_path = "./models/model_checkpoint.h5"
# model_weights_path = "./models/model_weights.h5"
# Define the number of epochs.
epochs = 10
batch_size = 32




load_data = Load_Data()
char_to_num= load_data.char_to_num
num_to_char  = load_data.num_to_char
df_train = load_data.df_train
df_val = load_data.df_val
wavs_path =load_data.wavs_path

prepro_data = Preporcess_Data(wavs_path,char_to_num)
fft_length = prepro_data.fft_length
# Define the training dataset
train_dataset = tf.data.Dataset.from_tensor_slices(
    (list(df_train["file_name"]), list(df_train["normalized_transcription"]))
)
train_dataset = (
    train_dataset.map(prepro_data.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Define the validation dataset
validation_dataset = tf.data.Dataset.from_tensor_slices(
    (list(df_val["file_name"]), list(df_val["normalized_transcription"]))
)
validation_dataset = (
    validation_dataset.map(prepro_data.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)


model = build_model(
    input_dim=fft_length // 2 + 1,
    output_dim=char_to_num.vocabulary_size(),
    loss = CTCLoss,
    rnn_units=512,
)
model.summary(line_length=110)


# Train the model


# Define checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=model_checkpoint_path,
    save_best_only=False,  # Save the model on every epoch
    save_weights_only=False,  # Save the entire model (including architecture)
    verbose=1
)



# Load the model if a checkpoint exists
if os.path.exists(model_checkpoint_path):
    # model = keras.models.load_model(model_checkpoint_path)
    custom_objects = {"CTCLoss": CTCLoss}
    with keras.utils.custom_object_scope(custom_objects):
        model = keras.models.load_model(model_checkpoint_path)





# Callback function to check transcription on the val set.
validation_callback = CallbackEval(validation_dataset,num_to_char,model,wer)



# Training loop
model.fit(
    train_dataset,
    epochs=epochs,
    callbacks=[checkpoint_callback, validation_callback],
    
)






