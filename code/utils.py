from tensorflow import keras
import numpy as np
import tensorflow as tf
import os, re

class CallbackEval(keras.callbacks.Callback):

    def __init__(self, dataset,num_to_char,model,wer,log_file):
        super().__init__()
        self.dataset = dataset
        self.num_to_char = num_to_char
        self.model = model
        self.wer = wer
        self.log_file = log_file
    
    def on_epoch_end(self, epoch: int, logs=None):
        predictions = []
        targets = []
        for batch in self.dataset:
            X, y = batch
            batch_predictions = self.model.predict(X)
            batch_predictions = decode_batch_predictions(batch_predictions,self.num_to_char)
            predictions.extend(batch_predictions)
            for label in y:
                label = (
                    tf.strings.reduce_join(self.num_to_char(label)).numpy().decode("utf-8")
                )
                targets.append(label)
        try:
            wer_score = self.wer(targets, predictions)
            wer_score = wer_score*100
            print("-" * 100)
            print(f"Word Error Rate: {wer_score:.8f}")
            print("-" * 100)
            for i in np.random.randint(0, len(predictions), 4):
                print(f"Target    : {targets[i]}")
                print(f"Prediction: {predictions[i]}")
                print("-" * 100)

            # Log metrics to file
            with open(self.log_file, 'a') as f:
                f.write(f"Epoch {epoch + 1} | WER {wer_score:.8f} | Loss {logs['loss']:.8f}\n")

        
        except Exception as e:
            print("-" * 100)
            print(f"In Except: {e}")
            for i in np.random.randint(0, len(predictions), 2):
                print(f"Target    : {targets[i]}")
                print(f"Prediction: {predictions[i]}")
                print("-" * 100)



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
def decode_batch_predictions(pred,num_to_char):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text


# def get_latest_checkpoint(directory = "../models/"):
#     # Get all filenames in the directory
#     filenames = os.listdir(directory)
#     # Define a regular expression pattern to extract numbers
#     pattern = re.compile(r'\((\d+)\)')
#     # Extract numbers from each filename and store in a list
#     numbers = [int(re.search(pattern, filename).group(1)) for filename in filenames if re.search(pattern, filename)]
#     # Find the maximum number
#     latest_checkpoint = max(numbers, default=None)
#     return latest_checkpoint



def get_latest_checkpoint(directory="../models/", pattern=None):
    # Get all filenames in the directory
    filenames = os.listdir(directory)
    # Filter filenames based on the pattern
    if pattern:
        filenames = [filename for filename in filenames if pattern in filename]
    # Define a regular expression pattern to extract numbers
    num_pattern = re.compile(r'\((\d+)\)')
    # Extract numbers from each filename and store in a list
    numbers = [int(re.search(num_pattern, filename).group(1)) for filename in filenames if re.search(num_pattern, filename)]
    # Find the maximum number
    latest_checkpoint = max(numbers, default=0)
    return latest_checkpoint