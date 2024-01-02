import shutil
import os
import pandas as pd
main_path = "Dataset/cv-corpus-14.0-2023-06-23/ur"
src = os.path.join(main_path,"wav_files")
dst = os.path.join(main_path,"limited_wav_files")

count = 0
def copy_audio(df,src, dst,thresh = 20000):
    global count
    self_count = 0
    for audio in df["path"]:
        audio = audio[:-4]+".wav"
        if count<thresh:
            if audio not in os.listdir(dst):
                shutil.copy(os.path.join(src, audio), os.path.join(dst, audio))
                count+=1
                self_count+=1
        else:
            df = df.iloc[:self_count,:]
            break
    print(count)
    return df

train_df = pd.read_csv(os.path.join(main_path,"train.tsv"),sep="\t")
copy_audio(train_df,src,dst)
print("Train_df count: ",count)
test_df = pd.read_csv(os.path.join(main_path,"test.tsv"),sep="\t")
copy_audio(test_df,src,dst)
print("Test_df count: ",count)
validated_df = pd.read_csv(os.path.join(main_path,"validated.tsv"),sep="\t")
validated_df = copy_audio(validated_df,src,dst)
#merge train and validated and test
main_df = pd.concat([train_df,test_df,validated_df])
main_df.to_csv(os.path.join(main_path,"final_main_dataset.tsv"),sep="\t",index=False)
print("Total voice: ",count,"total df length",len(main_df))
