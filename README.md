# Dual-Mic Speech Enhancement for Defence Environments

SIH 2026 project for DRDO Problem Statement ID 26052 (Dept of Defence
Production / IDEX, Category: Hardware, Theme: Smart Vehicles). Builds a
real-time, embedded speech enhancement pipeline for dual-mic/headset
setups in noisy defence environments (helicopters, drones, sirens, gun
shots, explosions, wind).

## What this does

Takes noisy speech from a primary mic (optionally a reference mic for
adaptive filtering) and outputs enhanced, cleaner speech - suitable for
radio comms in loud environments. Two ways to run it:

- **Torch path** - DeepFilterNet3 (pretrained, fine-tuned on defence
  noise) via PyTorch. Used for development/evaluation on a laptop.
- **ONNX path** - same fine-tuned model, exported to ONNX and wrapped
  in a pure numpy DSP layer. No torch/Rust dependency - this is the
  path that actually deploys to the Raspberry Pi.

## Results (synthetic_defence_test, 48 files)

| Metric          | Target      | ONNX wrapper result | Status |
|-----------------|-------------|----------------------|--------|
| SNR after       | >= 15 dB    | 14.09 dB             | FAIL (close) |
| SNR improvement | informational | +8.98 dB           | -      |
| STOI            | >= 0.85     | 0.9059               | PASS   |
| PESQ            | >= 2.5      | 2.556                | PASS   |
| Latency         | <= 30 ms/frame | 1.25 ms/frame      | PASS   |

Note: the problem statement's exact SNR wording ("SNR > 15dB") is
ambiguous between absolute output SNR and improvement-delta - both are
reported in the eval scripts rather than betting on one interpretation.

## Repo structure

```
configs/
  config.yaml               pipeline-level config (sample rate, model
                             base_dir, evaluation targets)
src/
  audio/                    wav loading, framing utilities
  model/
    nlms.py                 classical adaptive filter (currently unused
                             in the default pipeline - see below)
  df_onnx_dsp.py             numpy DSP wrapper around the ONNX exports:
                             STFT, ERB filterbank, feature normalization,
                             deep-filter coefficient fusion, iSTFT,
                             plus chunked processing for streaming
  pipeline.py                sequences everything: torch path
                             (enhance_dtln_stage/enhance_audio_file) and
                             ONNX path (enhance_onnx_stage/
                             enhance_audio_file_onnx)
demo/
  live_demo.py                file-mode and mic-mode demo (record, THEN
                               process, THEN play back). Supports both
                               engines via --onnx.
  live_stream_onnx.py          true continuous streaming demo: mic in,
                                enhanced audio out, ONNX path only.
  eval_onnx_wrapper.py          batch-evaluates the ONNX wrapper against
                                 synthetic_defence_test
  diagnose_alignment.py         debug script for STFT/iSTFT timing
                                 alignment issues
models/
  onnx_export/                enc.onnx / erb_dec.onnx / df_dec.onnx -
                               the fine-tuned model, exported
external/
  DeepFilterNet3_finetuned/   vendored checkpoint + config.ini for the
                               torch path
requirements.txt              full dev/training dependencies (torch etc)
requirements-pi.txt           minimal Pi deployment dependencies (no
                               torch/Rust)
```

## Setup (dev machine)

```
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

torch==2.6.0 and torchaudio==2.6.0 are pinned exactly - they must match
each other, and newer torchaudio removes a module DeepFilterNet needs.

## Setup (Raspberry Pi / deployment)

```
pip install -r requirements-pi.txt
```

No torch, no Rust build required - only onnxruntime (prebuilt aarch64
wheel) plus numpy/scipy. See HANDOFF.md for the exact file list and run
order.

## Usage

**File mode, torch engine, with metrics (needs a clean reference file):**
```
python demo/live_demo.py --primary noisy.wav --reference noisy.wav --clean clean.wav
```

**File mode, ONNX engine:**
```
python demo/live_demo.py --onnx --primary noisy.wav --reference noisy.wav --clean clean.wav
```

**Live mic, record-then-playback (ONNX engine):**
```
python demo/live_demo.py --mic --onnx --duration 5
```

**Live mic, true continuous streaming (ONNX engine only):**
```
python demo/live_stream_onnx.py --chunk-seconds 1.0
```
Use headphones - simultaneous mic + speaker on one device causes
feedback otherwise.

**Batch evaluation against the synthetic test set:**
```
python demo/eval_onnx_wrapper.py
```

## Known limitations

- **Single-mic prototype**: no physical second (reference) mic exists
  yet, so NLMS (`--nlms`) is off by default - tested with an identical
  primary/reference signal, it actively hurts quality (destructive with
  a non-independent reference). NLMS itself is validated separately on
  synthetic dual-signal data (~21dB improvement) and should be
  re-evaluated once real dual-mic hardware exists.
- **Small defence-noise pool**: only 63 raw noise clips across 6
  categories were available for fine-tuning - SNR improvement plateaus
  around 8-9dB, short of the 15dB target, likely close to what this
  pool can teach the model rather than an undertrained-model issue.
- **Domain mismatch on real recordings**: validated numbers are on a
  synthetic noisy/clean test set. Real recorded audio (e.g. genuinely
  stereo dual-channel recordings, lossy-codec voice notes) can look
  different enough to the model that it over-suppresses - flagged as a
  documented limitation, not yet fully resolved.
- **Chunked, not frame-by-frame, real-time processing**: the ONNX path
  processes independent multi-second chunks (GRU state resets at each
  chunk boundary) rather than true per-frame streaming. A custom
  streaming export with frame-by-frame state carry was built and
  validated for the GRU-state problem in isolation, but a separate
  convolution-buffering problem (3-frame time kernels needing real
  neighboring frames) made true single-frame streaming unreliable given
  the time available - kept as a documented reference/future option.
- **Latency numbers are laptop-CPU only** - not yet re-measured on real
  Pi hardware.
