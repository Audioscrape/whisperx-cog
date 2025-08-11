# WhisperX Cog for Audioscrape

Production-ready WhisperX model for podcast transcription with speaker diarization and embeddings extraction.

## Features

- **High-quality transcription** using WhisperX large-v3 model
- **Speaker diarization** with PyAnnote for multi-speaker podcasts
- **Speaker embeddings** extraction (256-dimensional vectors) for voice identification
- **Long-form audio support** up to 4 hours
- **Word-level timestamps** for precise alignment
- **Automatic language detection** or specify language code

## Deployment

This model is deployed on Replicate at: https://replicate.com/audioscrape/whisperx

## Usage

```python
import replicate

output = replicate.run(
    "audioscrape/whisperx:latest",
    input={
        "audio": "https://example.com/podcast.mp3",
        "huggingface_token": "hf_...",  # Required for speaker diarization
        "enable_diarization": True,
        "min_speakers": 2,
        "max_speakers": 4,
        "return_word_timestamps": True
    }
)

# Access speaker embeddings
if "speaker_embeddings" in output:
    for speaker, embedding in output["speaker_embeddings"].items():
        print(f"{speaker}: {len(embedding)}-dimensional vector")
```

## Output Format

```json
{
  "segments": [...],
  "speaker_embeddings": {
    "SPEAKER_00": [256-dimensional float array],
    "SPEAKER_01": [256-dimensional float array]
  },
  "metadata": {
    "duration_seconds": 600,
    "language": "en",
    "num_segments": 150,
    "num_speakers": 2,
    "speakers": ["SPEAKER_00", "SPEAKER_01"],
    "embeddings_dims": 256
  }
}
```

## Requirements

- HuggingFace token for PyAnnote speaker diarization models
- GPU with at least 8GB VRAM for optimal performance

## License

This project uses WhisperX and PyAnnote models. Please refer to their respective licenses.