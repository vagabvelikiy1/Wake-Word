import os
import numpy as np
import librosa

from src.features import extract_mfcc
from src.model import build_model

DATA_DIR = "data"
WAKE_WORD = "omar"

X, y = [], []

for folder in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    label = 1 if folder == WAKE_WORD else 0

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            audio, _ = librosa.load(
                os.path.join(folder_path, file), sr=16000
            )
            mfcc = extract_mfcc(audio)
            X.append(mfcc)
            y.append(label)

X = np.array(X)

# ⬇️ VERY IMPORTANT
X = X[..., np.newaxis]
y = np.array(y)

print("Training data shape:", X.shape)

model = build_model(X.shape[1:])
#model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)
model.fit(X, y, epochs=5, batch_size=1)

model.save("wake_word_model.h5")
