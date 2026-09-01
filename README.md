# DTLN Defence Speech Enhancement (Simulated Prototype)

AI-powered speech enhancement / noise cancellation pipeline for high-noise defence
environments, built for SIH 2026 (DRDO problem statement — dual-mic speech
enhancement, embedded/real-time requirement).

## Scope of this phase

This is a **laptop-only virtual/simulated prototype**. No Raspberry Pi or physical
mic hardware is used here — dual-mic input is simulated in software. The design
choices (model size, frame/hop timing, causal-only architecture) are made with the
eventual Raspberry Pi 4 deployment in mind, but that deployment is out of scope for
this repo's current phase.

## Pipeline

```
Input (file or simulated dual-mic) 
    -> DTLN (AI noise suppression, causal, streaming-capable)
    -> NLMS (classical adaptive filter, EC-side logic)
    -> Enhanced speech output
```

- **DTLN** — Dual-signal Transformation LSTM Network (Westhausen & Meyer, 2020).
  ~1.8M params, two stacked LSTM cores (128 units each), fully causal. Pretrained
  weights fine-tuned in two phases (see `src/training/`).
- **NLMS** — classical adaptive filter, implemented separately from the AI model.
  Kept in its own module (`src/model/nlms.py`) to mirror the team split: this is
  conceptually the "hardware/EC-side" stage even though it's written here for the
  simulation.

## Datasets

1. **VoiceBank-DEMAND** — baseline/general training (28 speakers train / 2 held
   out test, DEMAND's 10 noise categories, 0/5/10/15dB SNR).
2. **Custom synthetic defence-noise set** — clean speech (VoiceBank) mixed with
   gunshot/helicopter/drone/wind/siren noise (freesound.org), -5/0/5/10dB SNR.
   Used for fine-tuning/specialization.

## Evaluation targets (from official problem statement)

| Metric | Target |
|---|---|
| SNR improvement | > 15 dB |
| STOI | > 0.85 |
| PESQ | > 2.5 (PS states ">25" — a typo, since PESQ is bounded at 4.5; flagged to mentor) |
| End-to-end latency | ~20-30 ms |

## Repo structure

```
configs/            # single source of truth for hyperparameters
data/                # raw/ and processed/ are gitignored (large, regenerable)
src/
  model/             # dtln.py (AI), nlms.py (classical DSP)
  audio/             # I/O, resampling, framing
  training/          # phase 1 baseline, phase 2 fine-tune scripts
  evaluation/        # SNR / STOI / PESQ / latency metrics
  pipeline.py        # end-to-end orchestration
demo/                # laptop-only live demo (mic/file in -> enhanced out + viz)
checkpoints/         # gitignored — trained weights
```

## Status

Skeleton stage — no trained weights yet. See in-file docstrings in `src/` for what
each module will do; implementation proceeds module by module starting with
`src/audio/` (I/O + framing), since everything downstream depends on it.
