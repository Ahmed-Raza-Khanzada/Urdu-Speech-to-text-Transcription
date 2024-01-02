import os
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from load_dataset import Load_Data
from preprocess import Preporcess_Data
from model import build_model
from utils import CallbackEval,CTCLoss,get_latest_checkpoint
from jiwer import wer
import time
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard

# tf.debugging.experimental.enable_dump_debug_info("../tfdbg2_logs", tensor_debug_mode="FULL_HEALTH")



print("="*100)
start_time = time.time()

start_date_time = datetime.now()
print("Model started Training on this Date and time below")
# Print the start date and time
print(start_date_time.strftime("%Y-%m-%d %H:%M:%S"))
print()
print("-"*80)



# # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.66666666666)

# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))





# # Set GPU memory growth
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Restrict TensorFlow to allocate 7 GB of memory on the first GPU
#         tf.config.experimental.set_virtual_device_configuration(
#             gpus[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7168)]
#         )
#     except RuntimeError as e:
#         print(e)







gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)






tf.get_logger().setLevel('ERROR')


# model_weights_path = "./models/model_weights.h5"
# Define the number of epochs. 

version = 'v5'


# (15 + 1) BatchSize = 8 Done
last_epoch = get_latest_checkpoint(pattern="5v")
new_train_epoch = 500

epochs = new_train_epoch 

batch_size = 64 #8 #16 #32



# Define file paths for model checkpoints
model_checkpoint_path = f"../models/model_checkpoint_5v_({last_epoch}).h5"



load_data = Load_Data()
char_to_num= load_data.char_to_num
num_to_char  = load_data.num_to_char
df_train = load_data.df_train
df_val = load_data.df_val
wavs_path =load_data.wavs_path

prepro_data = Preporcess_Data(wavs_path,char_to_num)
fft_length = prepro_data.fft_length


# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()

# Open a strategy scope.
with strategy.scope():
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





# # Load the latest checkpoint if it exists
# latest_checkpoint = tf.train.latest_checkpoint('../models/')
# if latest_checkpoint:
#     print(f"Resuming training from checkpoint: {latest_checkpoint}................................................................")
#     model = keras.models.load_model(latest_checkpoint, custom_objects={"CTCLoss": CTCLoss})

# # ...
strategy = tf.distribute.MirroredStrategy()

# Open a strategy scope.
with strategy.scope():
    # Load the model if a checkpoint exists
    if os.path.exists(model_checkpoint_path):
        # model = keras.models.load_model(model_checkpoint_path)
        custom_objects = {"CTCLoss": CTCLoss}
        # with keras.utils.custom_object_scope(custom_objects):
        #     model = keras.models.load_model(model_checkpoint_path)
        m = f"Resuming training from checkpoint: {model_checkpoint_path}"
        model = keras.models.load_model(model_checkpoint_path, custom_objects=custom_objects)
    else:
        m = f"Creating a fresh model as no checkpoint found"
        model = build_model(
        input_dim=fft_length // 2 + 1,
        output_dim=char_to_num.vocabulary_size(),
        loss = CTCLoss,
        rnn_units=512,
    )






model.summary(line_length=110)

# Train the model

print("*"*45,"Training Model started","*"*45)
print("*"*35,m,"*"*35)



# Define checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath= "../models/model_checkpoint_5v_({epoch:02d}).h5", #model_checkpoint_path,
    save_best_only=False,  # Save the model on every epoch
    save_weights_only=False,  # Save the entire model (including architecture)
    verbose=1,
    save_format='tf'
)

log_file = f"../training_metrics_{version}.log"
# Callback function to check transcription on the val set.
validation_callback = CallbackEval(validation_dataset, num_to_char, model, wer,log_file)

# Define log directory for TensorBoard
log_dir = "../logs_v5/"  # Choose a suitable directory
# Create a TensorBoard callback
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Training loop
model.fit(
    train_dataset,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[checkpoint_callback, validation_callback, tensorboard_callback],
    initial_epoch=last_epoch,
    
)

print()
print("*"*45,"Training Model Ended","*"*45)
end_time = time.time()

# Calculate the total time in minutes
total_time_minutes = (end_time - start_time) / 60

print("Epochs: ",epochs)
print("Batch Size: ",batch_size)
print(f"Total Time Taken in minutes to train model: ", total_time_minutes)

current_date_time = datetime.now()
print("Model Start and End Training Date and time below")
print("Training Start Date: ",start_date_time.strftime("%Y-%m-%d %H:%M:%S"))
print()
print("Training End Date: ",current_date_time.strftime("%Y-%m-%d %H:%M:%S"))
print()
print("Note this is Server Time(where this model is running) so Don't confuse")


