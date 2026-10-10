import numpy as np
import librosa

def load_and_standardize_audio(file_path_or_buffer, target_sr=22050, target_duration=3.0):
    """
    Loads an audio file and standardizes it to mono, target_sr, and target_duration.
    Pads with silence if too short; center-crops if too long.
    """
    y, sr = librosa.load(file_path_or_buffer, sr=target_sr, mono=True)
    target_length = int(target_sr * target_duration)
    
    if len(y) < target_length:
        # Pad with silence equally at start and end
        pad_width = target_length - len(y)
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        y = np.pad(y, (pad_left, pad_right), mode="constant")
    elif len(y) > target_length:
        # Center crop
        start = (len(y) - target_length) // 2
        y = y[start : start + target_length]
        
    return y, sr

def compute_log_mel_spectrogram(y, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    """
    Computes a 2D Log-Mel Spectrogram.
    Returns:
      mel_db: 2D array of shape (n_mels, time_frames)
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=8000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def extract_explainability_features(y, sr=22050, hop_length=512):
    """
    Extracts acoustic explainability parameters:
      1. Pitch (F0 in Hz) tracking over time
      2. RMS Energy dynamics over time
      3. Spectral Centroid (timbral brightness/harshness)
      4. Summary metrics explaining 'Why' an emotion is classified
    """
    # 1. Pitch contour (Fundamental Frequency F0)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
        sr=sr, hop_length=hop_length
    )
    # Replace NaNs with 0 for unvoiced regions
    f0_clean = np.nan_to_num(f0, nan=0.0)
    voiced_f0 = f0[~np.isnan(f0)] if np.any(~np.isnan(f0)) else np.array([0.0])
    mean_pitch = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    pitch_variance = float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0.0

    # 2. RMS Energy (vocal loudness & power)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    mean_energy = float(np.mean(rms))
    max_energy = float(np.max(rms)) if len(rms) > 0 else 0.0

    # 3. Spectral Centroid (frequency center of mass / sharpness)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    mean_centroid = float(np.mean(centroid))

    # Time axes in seconds
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

    # 4. Human-readable explainability reasoning
    diagnostic_insights = []
    
    if mean_pitch > 190 and max_energy > 0.08:
        diagnostic_insights.append("High pitch range accompanied by sharp energy spikes strongly correlates with vocal strain typical of Anger.")
    elif mean_pitch > 170 and pitch_variance > 35:
        diagnostic_insights.append("Dynamic melodic pitch variability indicates elevated emotional modulation typical of Happiness.")
    elif mean_pitch < 140 or max_energy < 0.04:
        diagnostic_insights.append("Subdued pitch baseline and low acoustic energy profile align with vocal de-escalation typical of Sadness.")
    else:
        diagnostic_insights.append("Balanced vocal dynamics with moderate pitch and steady cadence.")

    return {
        "times": times,
        "f0": f0_clean,
        "rms": rms,
        "spectral_centroid": centroid,
        "metrics": {
            "mean_pitch_hz": round(mean_pitch, 1),
            "pitch_std_hz": round(pitch_variance, 1),
            "mean_energy": round(mean_energy, 4),
            "max_energy": round(max_energy, 4),
            "mean_centroid_hz": round(mean_centroid, 1)
        },
        "diagnostic_insights": diagnostic_insights
    }
