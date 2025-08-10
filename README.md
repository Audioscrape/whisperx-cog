# Audioscrape WhisperX Cog

Production-ready WhisperX wrapper for audio transcription with speaker embeddings support.

## Features
- 🎯 **Speaker Embeddings**: Extract and return speaker embedding vectors
- 🎙️ **Optimized for Podcasts**: Tuned for long-form conversational audio
- 👥 **Advanced Diarization**: Identify and track speakers across episodes
- 🚀 **Production Ready**: Battle-tested on thousands of hours of audio

## Why This Fork?
We needed speaker embeddings for cross-episode speaker identification, which wasn't available in existing Cog wrappers. This implementation adds that capability while optimizing for audio-specific use cases.

## Credits
- [WhisperX](https://github.com/m-bain/whisperX) - The amazing core transcription engine (MIT License)
- [victor-upmeet/whisperx-replicate](https://github.com/victor-upmeet/whisperx-replicate) - Inspiration for the Cog wrapper structure
- Built with ❤️by [Audioscrape](https://www.audioscrape.com) to make audio content (podcasts, earning calls, court hearings, personal audio, etc. ) searchable by everyone

## Usage
Used in production at Audioscrape for transcribing and analyzing audio. Deploy on [Blitzcompute](https://www.blitzcompute.com) or your own infrastructure.

## License
MIT - Same as WhisperX
