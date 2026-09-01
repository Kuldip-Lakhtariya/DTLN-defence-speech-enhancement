from src.pipeline import enhance_dtln_stage
from src.audio.io import save_audio_file  # check actual function name in src/audio/io.py if this errors

dtln_output, _ = enhance_dtln_stage(
    "path/to/your/test_audio.wav",
    "path/to/your/test_audio.wav"
)

save_audio_file(dtln_output, "path/to/enhanced_output.wav", "configs/config.yaml")