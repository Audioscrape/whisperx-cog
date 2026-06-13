# Audioscrape WhisperX Cog - Usage Guide

## Overview
Production-ready WhisperX Cog optimized for podcast transcription with speaker embeddings extraction. Handles everything from 5-minute news briefs to 4-hour episodes.

## Quick Start

### 1. Local Testing

```bash
# Install Cog
pip install cog

# Test with default audio (no embeddings)
python test_local.py

# Test with HuggingFace token (includes embeddings)
export HF_TOKEN=hf_your_token_here
python test_local.py

# Test with custom audio file
python test_local.py /path/to/podcast.mp3 $HF_TOKEN
```

### 2. Deploy to Replicate

```bash
# Login to Replicate (one-time)
cog login

# Deploy the model
./deploy.sh
```

### 3. Test on Replicate

```bash
# Set credentials
export REPLICATE_API_TOKEN=r8_your_token
export HF_TOKEN=hf_your_token

# Run test
./test_replicate.sh
```

### 4. Update Audioscrape

After successful deployment, update `compute_platform_client.rs`:

```rust
// Change from:
"model": "victor-upmeet/whisperx:latest"

// To:
"model": "audioscrape/whisperx:latest"
```

## API Parameters

### Required Parameters
- `audio`: Audio file (Path) - Supports up to 4 hours
- `huggingface_token`: HuggingFace token for speaker diarization

### Optional Parameters
- `min_speakers`: Minimum speakers (1-20, None=auto-detect)
- `max_speakers`: Maximum speakers (1-20, None=auto-detect)
- `language`: Language code (e.g., 'en', None=auto-detect)
- `batch_size`: Batch size for transcription (1-32, default=8, auto-reduced for long audio)
- `enable_diarization`: Enable speaker diarization (default=true)
- `return_word_timestamps`: Return word-level timestamps (default=true)

## Output Format

```json
{
  "segments": [
    {
      "text": "And so my fellow Americans",
      "start": 0.0,
      "end": 2.5,
      "speaker": "SPEAKER_00",
      "words": [
        {"word": "And", "start": 0.0, "end": 0.2},
        {"word": "so", "start": 0.2, "end": 0.4},
        ...
      ]
    }
  ],
  "metadata": {
    "duration_seconds": 11.0,
    "language": "en",
    "num_segments": 3,
    "num_speakers": 1,
    "speakers": ["SPEAKER_00"]
  },
  "speaker_embeddings": {
    "SPEAKER_00": [0.123, -0.456, ...],  // 192-dimensional normalized vector
    "SPEAKER_01": [0.789, 0.012, ...]
  }
}
```

## Speaker Embeddings

### What You Get
- 192-dimensional embedding vectors per speaker
- L2-normalized for cosine similarity
- Consistent across different audio segments
- Suitable for cross-episode speaker matching

### Usage for Speaker Matching
```python
import numpy as np

def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2)  # Already normalized

# Example: Check if two speakers are the same
similarity = cosine_similarity(
    embeddings["SPEAKER_00"],
    other_episode_embeddings["SPEAKER_02"]
)

if similarity > 0.92:  # Recommended threshold
    print("Same speaker detected across episodes!")
```

## Handling Different Podcast Types

### Solo Podcasts (1 speaker)
```bash
min_speakers=1
max_speakers=1
```

### Interview Style (2 speakers)
```bash
min_speakers=2
max_speakers=2
```

### Panel Discussions (3-6 speakers)
```bash
min_speakers=3
max_speakers=6
```

### Auto-detect (recommended)
```bash
# Don't specify min/max_speakers
# Let PyAnnote determine optimal clustering
```

## Performance Considerations

### Memory Usage
- Short episodes (<30 min): ~2GB GPU RAM
- Medium episodes (30-90 min): ~4GB GPU RAM
- Long episodes (2-4 hours): ~6-8GB GPU RAM
- Batch size auto-adjusts for long audio

### Processing Time
- Approximately 1:5 ratio (1 hour audio = 5 minutes processing)
- Diarization adds ~20% overhead
- Word alignment adds ~10% overhead

## Troubleshooting

### No Speaker Embeddings
**Issue**: Output doesn't contain `speaker_embeddings`

**Solutions**:
1. Ensure valid HuggingFace token is provided
2. Check that `enable_diarization=true`
3. Verify audio contains speech (not silence/music)
4. Check HF token has access to pyannote models

### Memory Errors
**Issue**: CUDA out of memory

**Solutions**:
1. Reduce `batch_size` (minimum=1)
2. Process shorter audio segments
3. Ensure GPU has at least 8GB VRAM

### Wrong Language Detection
**Issue**: Incorrect language detected

**Solutions**:
1. Explicitly set `language` parameter
2. Ensure audio has clear speech in first 30 seconds

### Too Many/Few Speakers
**Issue**: Incorrect speaker count

**Solutions**:
1. Set explicit `min_speakers` and `max_speakers`
2. For solo podcasts, use `min_speakers=1, max_speakers=1`
3. For panels, increase `max_speakers` up to 20

## Testing Checklist

### Before Deployment
- [ ] Test with short audio (< 1 minute)
- [ ] Test with medium podcast (30 minutes)
- [ ] Test with long podcast (2+ hours)
- [ ] Test with solo speaker
- [ ] Test with multiple speakers
- [ ] Verify embeddings are 192-dimensional
- [ ] Verify embeddings are normalized

### After Deployment
- [ ] Test via Replicate API
- [ ] Verify webhook callbacks work
- [ ] Test with production HF token
- [ ] Update compute_platform_client.rs
- [ ] Test end-to-end on staging

## Example: Full Pipeline Test

```bash
# 1. Local test with sample audio
export HF_TOKEN=hf_your_token
python test_local.py

# 2. Deploy to Replicate
cog login
./deploy.sh

# 3. Test on Replicate
export REPLICATE_API_TOKEN=r8_your_token
./test_replicate.sh

# 4. Test with real podcast
replicate run audioscrape/whisperx \
  audio=https://example.com/podcast.mp3 \
  huggingface_token=$HF_TOKEN \
  min_speakers=2 \
  max_speakers=4
```

## Support

For issues or questions:
- Check logs in prediction output
- Review [WhisperX documentation](https://github.com/m-bain/whisperx)
- File issues at [Audioscrape/whisperx-cog](https://github.com/Audioscrape/whisperx-cog)