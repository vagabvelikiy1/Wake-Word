import librosa
import numpy as np
from tensorflow.keras.models import load_model
from src.features import extract_mfcc

SR = 16000

def sliding_window_detect(audio_path, model_path, threshold=0.8):
    model = load_model(model_path)
    audio, _ = librosa.load(audio_path, sr=SR)

    step = int(0.25 * SR)
    window = SR

    for i in range(0, len(audio) - window, step):
        chunk = audio[i:i + window]
        mfcc = extract_mfcc(chunk)
        mfcc = mfcc[np.newaxis, ..., np.newaxis]

        prob = model.predict(mfcc, verbose=0)[0][0]
        if prob > threshold:
            print(f"Wake word detected at {i / SR:.2f}s (p={prob:.2f})")
