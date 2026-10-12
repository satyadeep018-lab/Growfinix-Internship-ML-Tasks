import io
import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 22050
TARGET_DURATION_SEC = 3.0
TARGET_NUM_SAMPLES = int(TARGET_SR * TARGET_DURATION_SEC)
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

def load_and_standardize_audio(audio_source):
    if isinstance(audio_source, (io.BytesIO, bytes)):
        y, sr = sf.read(audio_source)
    else:
        y, sr = librosa.load(audio_source, sr=TARGET_SR, mono=True)

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    if len(y) > TARGET_NUM_SAMPLES:
        y = y[:TARGET_NUM_SAMPLES]
    elif len(y) < TARGET_NUM_SAMPLES:
        pad_width = TARGET_NUM_SAMPLES - len(y)
        y = np.pad(y, (0, pad_width), mode='constant')

    return y.astype(np.float32), TARGET_SR

def compute_log_mel_spectrogram(y, sr=TARGET_SR):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def extract_explainability_features(y, sr=TARGET_SR):
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr, hop_length=HOP_LENGTH)
    f0_clean = np.nan_to_num(f0, nan=0.0)
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=HOP_LENGTH)

    mean_pitch = float(np.mean(f0_clean[f0_clean > 0])) if np.any(f0_clean > 0) else 140.0
    pitch_std = float(np.std(f0_clean[f0_clean > 0])) if np.any(f0_clean > 0) else 20.0
    mean_energy = float(np.mean(rms))
    mean_centroid = float(np.mean(centroid))

    if mean_energy > 0.08 and mean_centroid > 2400:
        heuristic_probs = [0.75, 0.15, 0.10]
    elif pitch_std > 35 or mean_pitch > 190:
        heuristic_probs = [0.15, 0.75, 0.10]
    else:
        heuristic_probs = [0.10, 0.15, 0.75]

    return {
        "times": times,
        "f0": f0_clean,
        "rms": rms,
        "centroid": centroid,
        "mean_pitch": mean_pitch,
        "pitch_std": pitch_std,
        "mean_energy": mean_energy,
        "mean_centroid": mean_centroid,
        "heuristic_probabilities": heuristic_probs
    }
