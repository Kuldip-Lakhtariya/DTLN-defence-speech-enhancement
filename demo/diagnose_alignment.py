"""
One-file diagnostic: measures whether enhanced audio is time-shifted
relative to the clean reference, using cross-correlation to find the
lag that maximizes alignment. Also checks the noisy/clean pair as a
baseline (expected near-zero lag, since synthetic_defence_test mixes
are generated aligned) to confirm any shift is introduced by the
wrapper pipeline itself, not present in the source data. Recomputes
SNR after correcting for the found lag to test whether misalignment
alone explains the near-0dB SNR results.
"""

import os
import sys

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import correlate

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from demo.live_demo import compute_snr_db
import src.df_onnx_dsp as wrap

TEST_FILE = "p232_006.wav"
NOISY_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "synthetic_defence_test", "noisy", TEST_FILE)
CLEAN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "synthetic_defence_test", "clean", TEST_FILE)
ONNX_DIR = os.path.join(PROJECT_ROOT, "models", "onnx_export")


def find_lag(reference, target, max_lag):
    n = min(len(reference), len(target))
    reference = reference[:n]
    target = target[:n]
    corr = correlate(target, reference, mode="full")
    lags = np.arange(-n + 1, n)
    window = (lags >= -max_lag) & (lags <= max_lag)
    corr = corr[window]
    lags = lags[window]
    best_lag = int(lags[np.argmax(corr)])
    return best_lag


def apply_lag(signal, lag):
    if lag > 0:
        return signal[lag:]
    elif lag < 0:
        return np.concatenate([np.zeros(-lag, dtype=signal.dtype), signal])
    return signal


def main():
    noisy, sr_noisy = sf.read(NOISY_PATH, dtype="float32")
    clean, sr_clean = sf.read(CLEAN_PATH, dtype="float32")
    if noisy.ndim > 1:
        noisy = noisy.mean(axis=1)
    if clean.ndim > 1:
        clean = clean.mean(axis=1)

    native_sr = sr_noisy
    print(f"native sample rate: {native_sr}")

    baseline_lag = find_lag(clean, noisy, max_lag=200)
    print(f"noisy vs clean baseline lag: {baseline_lag} samples ({1000 * baseline_lag / native_sr:.2f} ms)")

    erb_inv_fb = wrap.build_erb_inv_fb(wrap.ERB_WIDTHS)
    enc_session, erb_dec_session, df_dec_session = wrap.load_sessions(ONNX_DIR)

    noisy_model_sr = librosa.resample(noisy, orig_sr=native_sr, target_sr=wrap.SR)
    enhanced_model_sr = wrap.enhance_chunk(noisy_model_sr, enc_session, erb_dec_session, df_dec_session, erb_inv_fb)
    enhanced = librosa.resample(enhanced_model_sr, orig_sr=wrap.SR, target_sr=native_sr)

    n = min(len(clean), len(noisy), len(enhanced))
    clean_c = clean[:n]
    enhanced_c = enhanced[:n]

    snr_no_correction = compute_snr_db(clean_c, enhanced_c)
    print(f"SNR after, no lag correction: {snr_no_correction:.2f} dB")

    enhanced_lag = find_lag(clean_c, enhanced_c, max_lag=2000)
    print(f"enhanced vs clean lag: {enhanced_lag} samples ({1000 * enhanced_lag / native_sr:.2f} ms)")

    corrected = apply_lag(enhanced_c, enhanced_lag)
    n2 = min(len(clean_c), len(corrected))
    snr_corrected = compute_snr_db(clean_c[:n2], corrected[:n2])
    print(f"SNR after, WITH lag correction: {snr_corrected:.2f} dB")


if __name__ == "__main__":
    main()