import os
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from load_dataset import Load_Data
from preprocess import Preporcess_Data
from utils import decode_batch_predictions,CTCLoss

model_checkpoint_path = "./models/model_checkpoint.h5"

load_data = Load_Data()
char_to_num= load_data.char_to_num
num_to_char  = load_data.num_to_char
wavs_path =load_data.wavs_path


prepro_data = Preporcess_Data(wavs_path,char_to_num,predict=True)


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


def take_input():
    int1 = input("Enter audio file name:")
    return int1
# a = True
while True:
    wav_file  =  take_input()
    if wav_file == "":
        # a = False
        break
    # validation_dataset = tf.data.Dataset.from_tensor_slices(
    # wav_file )
 
    spectograms = prepro_data.encode_single_sample(wav_file)

    # Let's check results on more validation samples
    predictions = []
    #reshape spectograms (1,spectograms.shape[0],spectograms.shape[1])
    spectograms=tf.expand_dims(spectograms, axis=0)
    print(spectograms.shape,"@@@@@@@@@@@@@")
    # spectograms = tf.expand_dims(spectograms, axis=0)
    batch_predictions1 = model.predict(spectograms)
    # print(batch_predictions1[:2])
    batch_predictions = decode_batch_predictions(batch_predictions1,num_to_char)
    print(batch_predictions,len(batch_predictions))
    predictions.extend(batch_predictions)






