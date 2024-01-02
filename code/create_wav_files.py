import os
from pydub import AudioSegment

audio_path = "Dataset/cv-corpus-15.0-2023-09-08/ur/clips"
for pos,mp3_file in enumerate(os.listdir(audio_path)):
    if mp3_file.endswith(".mp3"):
        # Load the audio file

        audio = AudioSegment.from_mp3(os.path.join(audio_path,mp3_file))
        # Convert to WAV format
        wav_file = f"Dataset/cv-corpus-15.0-2023-09-08/ur/wav_files/{mp3_file[:-4]}.wav"
        audio.export(wav_file, format="wav")
        # print(mp3_file)
        # print("Done")
        # break
