import numpy as np
import librosa

SR, DURATION, N_MELS, HOP = 22050, 3.0, 64, 512

def load_mel(path_or_file):
    y, _ = librosa.load(path_or_file, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    n = int(SR * DURATION)
    y = np.pad(y, (0, max(0, n - len(y))))[:n]
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=HOP)
    return librosa.power_to_db(mel, ref=1.0, top_db=80.0).astype("float32")
