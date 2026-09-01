"""
TEMPORARY DIAGNOSTIC - not one of the 7 tracked files, do not commit as a
permanent part of the repo. Isolates which stage of the pipeline (DTLN
alone, vs DTLN+NLMS) is responsible for the negative SNR seen in
metrics.py, by reusing pipeline.py's exact same imports and logic but
stopping BEFORE apply_nlms() to get an intermediate DTLN-only output.

Run: python debug_pipeline_stages.py
"""

import numpy as np
import soundfile as sf

from src.audio.io import load_audio_file
from src.audio.framing import frame_signal, overlap_add
from src.model.dtln import load_pretrained_model, enhance_frame, reset_model_state
from src.model.nlms import apply_nlms

CONFIG_PATH = "configs/config.yaml"
CHECKPOINT = "./models_phase1_baseline/phase1_baseline.weights.h5"

CLEAN_PATH = "data/raw/clean_testset_wav/p232_002.wav"
NOISY_PATH = "data/raw/noisy_testset_wav/p232_002.wav"


def snr_db(clean, estimate):
    n = min(len(clean), len(estimate))
    clean = clean[:n]
    estimate = estimate[:n]
    noise = clean - estimate
    sp = np.sum(clean.astype(np.float64) ** 2)
    npow = np.sum(noise.astype(np.float64) ** 2)
    if npow < 1e-12:
        return 100.0
    if sp < 1e-12:
        return -100.0
    return 10.0 * np.log10(sp / npow)


def describe(name, arr):
    print(f"{name}: len={len(arr)} max_abs={np.max(np.abs(arr)):.4f} "
          f"mean_abs={np.mean(np.abs(arr)):.6f} has_nan={np.isnan(arr).any()} "
          f"has_inf={np.isinf(arr).any()}")


clean = load_audio_file(CLEAN_PATH, CONFIG_PATH)
noisy = load_audio_file(NOISY_PATH, CONFIG_PATH)

describe("clean", clean)
describe("noisy", noisy)
print(f"SNR(clean, noisy) [baseline, no processing]: {snr_db(clean, noisy):.2f} dB")

frames = frame_signal(noisy, CONFIG_PATH)
print(f"frames.shape = {frames.shape}")

model = load_pretrained_model(CHECKPOINT)
reset_model_state(model)

enhanced_frames = np.stack(
    [enhance_frame(model, frames[i]) for i in range(frames.shape[0])]
)
dtln_output = overlap_add(enhanced_frames, CONFIG_PATH)
dtln_output = dtln_output[: len(noisy)]

describe("dtln_output (before NLMS)", dtln_output)
print(f"SNR(clean, dtln_output) [DTLN alone]: {snr_db(clean, dtln_output):.2f} dB")

reference = noisy * 0.9  # rough stand-in for simulate_dual_mic's second channel
common_len = min(len(dtln_output), len(reference))
final_output = apply_nlms(dtln_output[:common_len], reference[:common_len])

describe("final_output (after NLMS)", final_output)
print(f"SNR(clean, final_output) [DTLN + NLMS]: {snr_db(clean, final_output):.2f} dB")