import librosa
import numpy as np

SR = 16000
SAMPLES = SR

def extract_mfcc(audio):
    if len(audio) < SAMPLES:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))
    else:
        audio = audio[:SAMPLES]

    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=13)
    return mfcc.T
