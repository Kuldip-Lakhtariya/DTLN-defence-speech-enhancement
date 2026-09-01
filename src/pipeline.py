"""
End-to-end speech enhancement pipeline: dual-mic input -> DeepFilterNet3
(pretrained AI noise suppression, run on the whole utterance) -> NLMS
(classical adaptive filtering using the reference mic) -> enhanced output.
This file only sequences src/audio, the DeepFilterNet3 model, and
src/model/nlms.py - no DSP or model math lives here beyond resampling glue.

ASSUMPTION (confirm with team/mentor): "reference_mic_path" is the noise-
reference mic on the SAME headset as the primary mic - not the other
person's mic on a call. Unchanged from the DTLN version of this file.

MODEL CHANGE NOTE (replaces DTLN): DTLN was not hitting target SNR/STOI/
PESQ after training. Swapped the enhancement stage to DeepFilterNet3
(Schroeter et al., pretrained, from github.com/Rikorose/DeepFilterNet).

BUGFIX NOTE: _get_deepfilter_model() previously hardcoded
model_base_dir="external/DeepFilterNet3" directly in code, ignoring
configs/config.yaml's model.base_dir field entirely - editing that yaml
field did nothing. Fixed to read base_dir from config_path so config.yaml
is the actual single source of truth, matching its own stated rationale.
Now points at external/DeepFilterNet3_finetuned (fine-tuned checkpoint,
model_132.ckpt.best) instead of the original zero-shot pretrained one.

FLAG (process-level caching): _model and _df_state are cached in module-
level globals and only loaded once per process. If you ever need to run
enhance_audio_file() against two different base_dir values in the SAME
Python process (e.g. comparing pretrained vs fine-tuned in one script),
this cache will silently keep serving the first one loaded. Not an issue
for metrics.py run as a single process against one checkpoint, but flag
this if you ever build an A/B comparison script.

FLAG 1 (Pi, no internet): pre-download the checkpoint zip once, while you
have internet, from:
  https://github.com/Rikorose/DeepFilterNet/raw/main/models/DeepFilterNet3.zip
Unzip it into the DeepFilterNet cache dir on the Pi (appdirs cache path,
typically ~/.cache/DeepFilterNet3/), or vendor it into external/ like
external/DTLN/ and point model_base_dir at it directly. Confirmed real
size: 7.9MB zip, 8.7MB checkpoint file - trivial for 4GB RAM. Note: if
deploying the fine-tuned checkpoint to Pi instead of pretrained, vendor
external/DeepFilterNet3_finetuned/ the same way (8.3MB, same order of
size).

FLAG 2 (sample rate): DeepFilterNet3 is trained at 48kHz (confirmed from
its config.ini), not your pipeline's 16kHz. This file resamples primary
audio 16kHz -> 48kHz before the model and back to 16kHz -> after, so NLMS
and everything downstream still runs at your original 16kHz. Extra
resampling cost is small relative to the model itself but is a new
CPU cost that did not exist with DTLN - worth timing on the Pi.

FLAG 3 (export target): configs/config.yaml still says "TFLite int8" for
export. That target no longer applies - DeepFilterNet3 is PyTorch, not
TF/Keras. Do not act on this yet; revisit only when you actually reach
Raspberry Pi deployment (ONNX + ONNX Runtime is the likely replacement
path, not TFLite).

COMMIT NOTE: fixed _get_deepfilter_model() to read model_base_dir from
config.yaml instead of a hardcoded string; config.yaml model.base_dir
switched to external/DeepFilterNet3_finetuned to evaluate the fine-tuned
checkpoint (model_132.ckpt.best) instead of the zero-shot pretrained one.
"""

import numpy as np
import torch
import yaml
from scipy.signal import resample_poly

from src.audio.io import load_audio_file
from src.model.nlms import apply_nlms

DEEPFILTER_SAMPLE_RATE = 48000

_model = None
_df_state = None


def _load_pipeline_sample_rate(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return int(config["audio"]["sample_rate"])


def _load_model_base_dir(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["model"]["base_dir"]


def _get_deepfilter_model(config_path):
    global _model, _df_state
    if _model is None:
        base_dir = _load_model_base_dir(config_path)
        from df.enhance import init_df
        _model, _df_state, _ = init_df(model_base_dir=base_dir)
    return _model, _df_state


def _resample(signal, orig_sr, target_sr):
    if orig_sr == target_sr:
        return signal
    return resample_poly(signal, target_sr, orig_sr).astype(np.float32)


def enhance_dtln_stage(primary_mic_path: str, reference_mic_path: str,
                        config_path: str = "configs/config.yaml"):
    from df.enhance import enhance

    pipeline_sr = _load_pipeline_sample_rate(config_path)
    primary_signal = load_audio_file(primary_mic_path, config_path)
    reference_signal = load_audio_file(reference_mic_path, config_path)

    model, df_state = _get_deepfilter_model(config_path)

    primary_48k = _resample(primary_signal, pipeline_sr, DEEPFILTER_SAMPLE_RATE)
    primary_tensor = torch.from_numpy(primary_48k).unsqueeze(0)

    try:
        enhanced_48k = enhance(model, df_state, primary_tensor)
    except Exception as exc:
        raise RuntimeError(f"DeepFilterNet3 enhancement failed: {exc}") from exc

    enhanced_48k = enhanced_48k.squeeze(0).numpy()
    dtln_output = _resample(enhanced_48k, DEEPFILTER_SAMPLE_RATE, pipeline_sr)

    dtln_output = dtln_output[: len(primary_signal)]

    common_length = min(len(dtln_output), len(reference_signal))
    if common_length == 0:
        raise ValueError("primary or reference signal is empty after loading")
    dtln_output = dtln_output[:common_length]
    reference_signal = reference_signal[:common_length]

    return dtln_output, reference_signal


def enhance_audio_file(primary_mic_path: str, reference_mic_path: str,
                        config_path: str = "configs/config.yaml",
                        use_nlms: bool = False) -> np.ndarray:
    dtln_output, reference_signal = enhance_dtln_stage(
        primary_mic_path, reference_mic_path, config_path
    )
    if use_nlms:
        return apply_nlms(dtln_output, reference_signal)
    return dtln_output