from tensorflow import keras
import os
import pandas as pd
import pickle

class Load_Data():
    def __init__(self,data_path="./",out_path="./"):
        self.out_path = out_path
        self.data_path=data_path#if path not  changed then /content/drive/MyDrive/Urdu_Speech_wavs/
        self.wavs_path = self.data_path + "/Dataset/cv-corpus-14.0-2023-06-23/ur/limited_wav_files/"
        self.metadata_path = self.data_path + "/Dataset/cv-corpus-14.0-2023-06-23/ur/final_main_dataset.tsv"
        self.vocab_path_inp = self.data_path + "/Dataset/char_to_num_vocab_v2.pkl"
        self.vocab_path_out = self.out_path + "/Dataset/char_to_num_vocab_v2.pkl"

        # Read metadata file and parse it
        self.metadata_df = pd.read_csv(self.metadata_path, sep="\t")
        # self.metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
        self.metadata_df = self.metadata_df[["path", "sentence"]]
        self.new_column_names = {'path': 'file_name', 'sentence':"normalized_transcription"}

        if os.path.exists(self.vocab_path_inp):
            loaded_vocab = None
            with open(self.vocab_path_inp, "rb") as f:
                loaded_vocab = pickle.load(f)
               
            # Creating the integer to character mapping using the loaded vocabulary
            self.char_to_num = keras.layers.StringLookup(vocabulary=loaded_vocab, oov_token="")
            self.num_to_char = keras.layers.StringLookup(vocabulary=loaded_vocab, oov_token="", invert=True)
        else:
            chars_vocab = {'’', 'ى', 'ُ', '”', 'ﷲ', 'ﷺ', 'ض', 'ؤ', '؟', 'ظ', 'ن', 'ﻧ', 'ﺗ', 'و', 'ؓ', 'ه', '`', 'ً', 'ﯾ', 'د', 'ؔ', 'ْ', 'ٰ', 'ﺲ', 'ل', 'ت', '"', 'ش', 'ی', ':', 'ک', "'", 'ء', 'م', 'ٔ', 'ہ', 'ے', 'ژ', 'ۂ', 'ِ', 'ح', 'گ', 'ﺩ', 'چ', 'ص', 'ڈ', 'ﭨ', 'ك', '“', 'ٓ', 'ٗ', ',', 'ي', 'پ', 'ڑ', 'ث', 'َ', 'ف', 'ﮯ', 'ع', 'ب', 'آ', 'ر', '۔', 'ا', 'ھ', 'س', 'ئ', 'ذ', '.', 'أ', '!', 'ز', 'ط', 'خ', 'ﮭ', '،', 'ٹ', 'ۃ', '-', 'ج', 'ﺘ', 'ق', '…', ' ', 'غ', 'ؑ', 'ۓ', '؛', 'ﻮ', '‘', 'ں', 'ّ'}
            characters = list(chars_vocab)
            allow = ["?", " ", "!", "'"]
            alphas = "abcdefghijklmnopqrstuvwxyz"
            characters = [i for i in "".join(characters) if ((i.isalpha()) or (i in allow)) and (i not in alphas)]

            # Creating the character to integer mapping
            self.char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
            # Saving the vocabulary using pickle
            # with open(self.vocab_path_out, "wb") as f:
            #     pickle.dump(self.char_to_num.get_vocabulary(), f)
            self.num_to_char =  keras.layers.StringLookup(vocabulary=self.char_to_num.get_vocabulary(), oov_token="", invert=True)
        print(
            f"The loaded vocabulary is: {self.num_to_char.get_vocabulary()} "
            f"(size ={self.num_to_char.vocabulary_size()})"
        )
        print(
            f"The loaded vocabulary is: {self.char_to_num.get_vocabulary()} "
            f"(size ={self.char_to_num.vocabulary_size()})")
        self.__clean_dataset__()
        self.__train_test_split__()
    def __clean_dataset__(self):
        # Rename columns using the .rename() method
        self.metadata_df =self.metadata_df.rename(columns=self.new_column_names)
        self.metadata_df = self.metadata_df.sample(frac=1).reset_index(drop=True)
        # self.metadata_df = self.metadata_df.iloc[:500,:]
        self.metadata_df["file_name"] = self.metadata_df["file_name"].apply(lambda x: x[:-4])
        self.metadata_df = self.metadata_df[self.metadata_df['normalized_transcription'].str.len() <= 200]
        #drop everything row which has doplicates normalized_transcription
        self.metadata_df = self.metadata_df.drop_duplicates(subset=['normalized_transcription'])
    def __train_test_split__(self,train_size=0.90):
        split = int(len(self.metadata_df) * train_size)
        self.df_train = self.metadata_df[:split]
        self.df_val = self.metadata_df[split:]

        print(f"Size of the training set: {len(self.df_train)}")
        print(f"Size of the training set: {len(self.df_val)}")
    def show_max_sentence(self):
        maxl = 0
        max_se = ""
        pos1 = None
        for pos,i in enumerate(self.metadata_df["normalized_transcription"]):
            if len(i)>maxl:
                maxl = len(i)
                max_se = i
                pos1 = pos
            print(maxl,max_se)
            # print(self.metadata_df.head(3))
            # print(pos1)
    
    def check_head(self):
        print(self.metadata_df.head())