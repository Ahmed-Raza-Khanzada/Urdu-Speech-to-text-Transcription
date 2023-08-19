class Preporcess_Data()
    def __init__(self,wavs_path,char_to_num,frame_length = 256,frame_step  =160,fft_length = 384) -> None:
        # An integer scalar Tensor. The window length in samples.
        self.frame_length = frame_length#600#256
        # An integer scalar Tensor. The number of samples to step.
        self.frame_step = frame_step#307#160
        # An integer scalar Tensor. The size of the FFT to apply.
        # If not provided, uses the smallest power of 2 enclosing frame_length.
        self.fft_length = fft_length#650#384
        self.char_to_num  = char_to_num
        self.wavs_path = wavs_path
    def encode_single_sample(wav_file, label,predict = False):
        ###########################################
        ##  Process the Audio
        ##########################################
        # 1. Read wav

        file = tf.io.read_file(self.wavs_path + wav_file+".wav")
        # 2. Decode the wav file
        audio, _ = tf.audio.decode_wav(file)
        audio = tf.squeeze(audio, axis=-1)
        # 3. Change type to float
        audio = tf.cast(audio, tf.float32)
        # 4. Get the spectrogram
        spectrogram = tf.signal.stft(
            audio, frame_length=self.frame_length, frame_step=self.frame_step, fft_length=self.fft_length
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
        if not predict:
            label = tf.strings.lower(label)
            # 8. Split the label
            label = tf.strings.unicode_split(label, input_encoding="UTF-8")
            # 9. Map the characters in label to numbers
            label = self.char_to_num(label)
            # 10. Return a dict as our model is expecting two inputs
            return spectrogram, label
        return spectrogram
