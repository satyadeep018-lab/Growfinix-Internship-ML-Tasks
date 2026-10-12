import os
import argparse
import numpy as np
import soundfile as sf
from acoustic_features import TARGET_SR, TARGET_DURATION_SEC

def synthesize_audio_samples(output_dir="data/samples"):
    os.makedirs(output_dir, exist_ok=True)
    t = np.linspace(0, TARGET_DURATION_SEC, int(TARGET_SR * TARGET_DURATION_SEC), endpoint=False)

    # Angry sample (High energy, harsh timbre)
    y_angry = 0.5 * np.sin(2 * np.pi * 240 * t) + 0.3 * np.sin(2 * np.pi * 480 * t) + 0.1 * np.random.normal(0, 0.05, len(t))
    sf.write(os.path.join(output_dir, "sample_angry.wav"), y_angry.astype(np.float32), TARGET_SR)

    # Happy sample (Melodic pitch vibrato)
    f_happy = 210 + 35 * np.sin(2 * np.pi * 4 * t)
    phase_happy = 2 * np.pi * np.cumsum(f_happy) / TARGET_SR
    y_happy = 0.4 * np.sin(phase_happy)
    sf.write(os.path.join(output_dir, "sample_happy.wav"), y_happy.astype(np.float32), TARGET_SR)

    # Sad sample (Low pitch, soft amplitude decay)
    y_sad = 0.25 * np.sin(2 * np.pi * 125 * t) * np.linspace(1.0, 0.4, len(t))
    sf.write(os.path.join(output_dir, "sample_sad.wav"), y_sad.astype(np.float32), TARGET_SR)
    print(f"Generated synthetic benchmark samples in '{output_dir}'.")

if __name__ == "__main__":
    synthesize_audio_samples()
