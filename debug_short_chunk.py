"""
TEMPORARY DIAGNOSTIC - not one of the 7 tracked files, do not commit.
Tests the SAME official batch model on a short ~1s slice of the test file
(matching training.phase1_baseline.chunk_length_seconds) vs the full file,
to check whether performance is healthy on training-length input but
degrades on much longer real files (LSTM long-sequence drift hypothesis).

Run: python debug_short_chunk.py
"""

import sys
import os
import numpy as np
import soundfile as sf

_EXTERNAL_DTLN_PATH = os.path.abspath("external/DTLN")
sys.path.append(_EXTERNAL_DTLN_PATH)
from DTLN_model import DTLN_model as _OfficialDTLNModel

CHECKPOINT = "./models_phase1_baseline/phase1_baseline.weights.h5"
CLEAN_PATH = "data/raw/clean_testset_wav/p232_002.wav"
NOISY_PATH = "data/raw/noisy_testset_wav/p232_002.wav"

CHUNK_SECONDS = 1.0
FS = 16000
CHUNK_SAMPLES = int(FS * CHUNK_SECONDS)


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


clean, _ = sf.read(CLEAN_PATH)
noisy, _ = sf.read(NOISY_PATH)
clean = clean.astype(np.float32)
noisy = noisy.astype(np.float32)

clean_short = clean[:CHUNK_SAMPLES]
noisy_short = noisy[:CHUNK_SAMPLES]

dtln_wrapper = _OfficialDTLNModel()
dtln_wrapper.build_DTLN_model()
dtln_wrapper.model.load_weights(CHECKPOINT)

short_input = noisy_short.reshape(1, -1).astype(np.float32)
short_output = np.array(dtln_wrapper.model(short_input, training=False)).reshape(-1)

full_input = noisy.reshape(1, -1).astype(np.float32)
full_output = np.array(dtln_wrapper.model(full_input, training=False)).reshape(-1)

print(f"Full file length: {len(noisy)} samples (~{len(noisy)/FS:.1f}s, "
      f"~{1 + (len(noisy) - 512)//128} frames)")
print(f"Short chunk length: {len(noisy_short)} samples (~{CHUNK_SECONDS}s, "
      f"~{1 + (CHUNK_SAMPLES - 512)//128} frames, matching training)")
print()
print(f"SNR(clean_short, noisy_short) [baseline, short]: "
      f"{snr_db(clean_short, noisy_short):.2f} dB")
print(f"SNR(clean_short, short_output) [model on SHORT training-length input]: "
      f"{snr_db(clean_short, short_output):.2f} dB")
print()
print(f"SNR(clean, noisy) [baseline, full]: {snr_db(clean, noisy):.2f} dB")
print(f"SNR(clean, full_output) [model on FULL 8s input]: "
      f"{snr_db(clean, full_output):.2f} dB")