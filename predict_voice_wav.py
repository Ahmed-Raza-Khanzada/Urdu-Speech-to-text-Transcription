import os
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from Scripts.load_dataset import Load_Data
from Scripts.preprocess import Preporcess_Data
from Scripts.utils import decode_batch_predictions,CTCLoss

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
# app = Flask('flaskapp', static_url_path='/static110', static_folder='static110')
# app = Flask(__name__, static_url_path='/static', static_folder='/static')
CORS(app)

# app.static_folder = '../static'

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_checkpoint_path = "./models/model_checkpoint_v2.h5"

load_data = Load_Data(data_path="./",out_path="./",predict=True)
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
else:
    print("Model not found. Please train or download model...")


# def take_input():
#     int1 = input("Enter audio file name:")
#     return int1
# # a = True
# while True:
#     wav_file  =  take_input()
#     if wav_file == "":
#         # a = False
#         break
#     # validation_dataset = tf.data.Dataset.from_tensor_slices(
#     # wav_file )

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    ARGs:
        audio_file       (file): The audio file to transcribe.
    Returns:
    """
    # Check if the 'audio_file' field exists in the request
    try:
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        # Get the uploaded audio file
        wav_file = request.files['audio_file']

        # Check if the file is empty
        if wav_file.filename == '':
            return jsonify({'error': 'Empty audio file'}), 400

        # Generate a safe filename for the uploaded file
        filename = secure_filename(wav_file.filename)

        # Save the uploaded file to the temporary directory
        wav_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        spectograms = prepro_data.encode_single_sample(filename)

        # Let's check results on more validation samples
        predictions = []
        #reshape spectograms (1,spectograms.shape[0],spectograms.shape[1])
        spectograms=tf.expand_dims(spectograms, axis=0)
        print("Spectogram shape: ",spectograms.shape)
        # spectograms = tf.expand_dims(spectograms, axis=0)
        batch_predictions1 = model.predict(spectograms)
        # print(batch_predictions1[:2])
        batch_predictions = decode_batch_predictions(batch_predictions1,num_to_char)
        print(batch_predictions[0][::-1])
        print(batch_predictions[0])
        predictions.extend(batch_predictions)

        return jsonify({'successfull':True,
                        'predictions': batch_predictions[0][::-1]}), 200
    except Exception as e:
        return jsonify({'error': str(e), 'successfull':False}), 400
#common_voice_ur_26562732,common_voice_ur_26562733


if __name__ == '__main__':
    # mainpath = os.path.abspath(os.getcwd())
    app.run(host='127.0.0.1', port=8001, debug=False)
