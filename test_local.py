#!/usr/bin/env python3
"""
Local test script for Audioscrape WhisperX Cog
Tests transcription with speaker embeddings extraction
"""

import json
import sys
import os
from pathlib import Path as PathLib
import tempfile
import urllib.request

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict import Predictor
from cog import Path


def download_test_audio():
    """Download a short test audio file if needed."""
    test_url = "https://github.com/openai/whisper/raw/main/tests/jfk.flac"
    test_file = "/tmp/test_audio.flac"
    
    if not os.path.exists(test_file):
        print(f"Downloading test audio from {test_url}...")
        urllib.request.urlretrieve(test_url, test_file)
        print(f"Downloaded to {test_file}")
    
    return test_file


def test_transcription(audio_path: str, hf_token: str = None):
    """Test the transcription with speaker embeddings."""
    
    print(f"\n🎯 Testing Audioscrape WhisperX Cog")
    print(f"📁 Audio file: {audio_path}")
    print("=" * 60)
    
    # Initialize predictor
    print("\n⚙️  Initializing predictor...")
    predictor = Predictor()
    predictor.setup()
    
    # Test parameters
    params = {
        "audio": Path(audio_path),
        "min_speakers": None,  # Auto-detect
        "max_speakers": None,  # Auto-detect
        "language": None,  # Auto-detect
        "huggingface_token": hf_token or "dummy_token",  # Will fail diarization if invalid
        "batch_size": 8,
        "enable_diarization": True if hf_token else False,
        "return_word_timestamps": True,
    }
    
    # Run prediction
    print("\n🚀 Running transcription...")
    print(f"   Diarization: {'✅ Enabled' if params['enable_diarization'] else '❌ Disabled'}")
    print(f"   Word timestamps: {'✅ Enabled' if params['return_word_timestamps'] else '❌ Disabled'}")
    
    try:
        result = predictor.predict(**params)
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Display results
    print("\n📊 Results:")
    print("=" * 60)
    
    # Metadata
    metadata = result.get("metadata", {})
    print(f"Language: {metadata.get('language', 'unknown')}")
    print(f"Duration: {metadata.get('duration_seconds', 0):.1f} seconds")
    print(f"Segments: {metadata.get('num_segments', 0)}")
    print(f"Speakers: {metadata.get('num_speakers', 0)}")
    
    if metadata.get('speakers'):
        print(f"Speaker labels: {', '.join(metadata['speakers'])}")
    
    # Check for speaker embeddings
    if "speaker_embeddings" in result and result["speaker_embeddings"]:
        print(f"\n🎯 Speaker Embeddings:")
        for speaker, embedding in result["speaker_embeddings"].items():
            print(f"  • {speaker}: {len(embedding)}-dimensional vector")
            # Show first 5 values as sample
            sample = embedding[:5] if len(embedding) >= 5 else embedding
            sample_str = ", ".join(f"{v:.4f}" for v in sample)
            print(f"    Sample: [{sample_str}, ...]")
    else:
        print("\n⚠️  No speaker embeddings extracted")
        if not hf_token:
            print("    (Diarization disabled - provide HuggingFace token to enable)")
    
    # Display sample segments
    segments = result.get("segments", [])
    if segments:
        print(f"\n📝 Sample Segments (first 3):")
        for i, segment in enumerate(segments[:3]):
            speaker = segment.get("speaker", "Unknown")
            text = segment.get("text", "").strip()
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            
            print(f"\n  [{i+1}] {speaker} ({start:.2f}s - {end:.2f}s):")
            print(f"      \"{text}\"")
            
            # Show word timestamps if available
            if "words" in segment and segment["words"]:
                words_sample = segment["words"][:5]
                words_str = " ".join(w.get("word", "") for w in words_sample)
                print(f"      Words: {words_str}...")
    
    # Save full output
    output_file = "/tmp/whisperx_test_output.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Full output saved to: {output_file}")
    
    # Summary
    print("\n" + "=" * 60)
    if "speaker_embeddings" in result and result["speaker_embeddings"]:
        print("✅ Test PASSED - Transcription with speaker embeddings successful!")
    else:
        print("⚠️  Test PARTIAL - Transcription successful but no embeddings")
        if not hf_token:
            print("    To test embeddings, run with: HF_TOKEN=your_token python test_local.py")


def main():
    """Main test function."""
    
    # Parse arguments
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        # Use default test audio
        audio_file = download_test_audio()
    
    # Get HuggingFace token
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    
    if len(sys.argv) > 2:
        hf_token = sys.argv[2]
    
    # Check audio file exists
    if not os.path.exists(audio_file):
        print(f"❌ Error: Audio file not found: {audio_file}")
        print("\nUsage:")
        print("  python test_local.py [audio_file] [hf_token]")
        print("\nOr set environment variable:")
        print("  export HF_TOKEN=your_token")
        print("  python test_local.py")
        sys.exit(1)
    
    if not hf_token:
        print("⚠️  Warning: No HuggingFace token provided")
        print("  Diarization and speaker embeddings will be disabled")
        print("  Set HF_TOKEN environment variable or pass as argument")
        response = input("\nContinue without diarization? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Run test
    test_transcription(audio_file, hf_token)


if __name__ == "__main__":
    main()