#!/bin/bash
# Test Audioscrape WhisperX on Replicate

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🧪 Testing Audioscrape WhisperX on Replicate${NC}"
echo "============================================"

# Check environment variables
if [ -z "$REPLICATE_API_TOKEN" ]; then
    echo -e "${RED}❌ Error: REPLICATE_API_TOKEN not set${NC}"
    echo "Export your token: export REPLICATE_API_TOKEN=r8_..."
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}⚠️  Warning: HF_TOKEN not set${NC}"
    echo "Diarization will fail without a valid HuggingFace token"
    echo "Export your token: export HF_TOKEN=hf_..."
    read -p "Continue without HF_TOKEN? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    HF_TOKEN="dummy_token"
fi

# Test audio URL (short sample)
AUDIO_URL="https://github.com/openai/whisper/raw/main/tests/jfk.flac"

echo ""
echo "📤 Submitting prediction to Replicate..."
echo "Audio: $AUDIO_URL"

# Create prediction
RESPONSE=$(curl -s -X POST \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "version": "latest",
    "input": {
      "audio": "'$AUDIO_URL'",
      "huggingface_token": "'$HF_TOKEN'",
      "enable_diarization": true,
      "return_word_timestamps": true,
      "min_speakers": null,
      "max_speakers": null,
      "language": null,
      "batch_size": 8
    }
  }' \
  https://api.replicate.com/v1/models/audioscrape/whisperx/predictions)

# Extract prediction ID
PREDICTION_ID=$(echo $RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin).get('id', ''))")

if [ -z "$PREDICTION_ID" ]; then
    echo -e "${RED}❌ Failed to create prediction${NC}"
    echo "Response: $RESPONSE"
    exit 1
fi

echo -e "${GREEN}✅ Prediction created: $PREDICTION_ID${NC}"
echo ""
echo "⏳ Waiting for completion..."

# Poll for completion
ATTEMPTS=0
MAX_ATTEMPTS=60  # 5 minutes max

while [ $ATTEMPTS -lt $MAX_ATTEMPTS ]; do
    sleep 5
    
    STATUS_RESPONSE=$(curl -s \
      -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
      https://api.replicate.com/v1/predictions/$PREDICTION_ID)
    
    STATUS=$(echo $STATUS_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', ''))")
    
    if [ "$STATUS" = "succeeded" ]; then
        echo -e "${GREEN}✅ Prediction succeeded!${NC}"
        
        # Extract and display results
        OUTPUT=$(echo $STATUS_RESPONSE | python3 -c "
import sys, json
data = json.load(sys.stdin)
output = data.get('output', {})
print(json.dumps(output, indent=2))
")
        
        echo ""
        echo "📊 Results:"
        echo "=========="
        
        # Parse and display key information
        python3 - <<EOF
import json
output = json.loads('''$OUTPUT''')

# Metadata
metadata = output.get('metadata', {})
print(f"Language: {metadata.get('language', 'unknown')}")
print(f"Duration: {metadata.get('duration_seconds', 0):.1f} seconds")
print(f"Segments: {metadata.get('num_segments', 0)}")
print(f"Speakers: {metadata.get('num_speakers', 0)}")

if metadata.get('speakers'):
    print(f"Speaker labels: {', '.join(metadata['speakers'])}")

# Speaker embeddings
embeddings = output.get('speaker_embeddings', {})
if embeddings:
    print(f"\n🎯 Speaker Embeddings:")
    for speaker, embedding in embeddings.items():
        print(f"  • {speaker}: {len(embedding)}-dimensional vector")
        sample = embedding[:3]
        sample_str = ", ".join(f"{v:.4f}" for v in sample)
        print(f"    Sample: [{sample_str}, ...]")
else:
    print("\n⚠️  No speaker embeddings found")

# Sample segments
segments = output.get('segments', [])
if segments:
    print(f"\n📝 Sample Segments (first 2):")
    for i, seg in enumerate(segments[:2]):
        speaker = seg.get('speaker', 'Unknown')
        text = seg.get('text', '').strip()
        print(f"  [{i+1}] {speaker}: \"{text}\"")
EOF
        
        # Save full output
        echo "$OUTPUT" > /tmp/replicate_test_output.json
        echo ""
        echo "💾 Full output saved to: /tmp/replicate_test_output.json"
        
        # Check if embeddings were extracted
        if echo "$OUTPUT" | grep -q "speaker_embeddings"; then
            echo ""
            echo -e "${GREEN}✅ TEST PASSED - Speaker embeddings successfully extracted!${NC}"
        else
            echo ""
            echo -e "${YELLOW}⚠️  TEST PARTIAL - No speaker embeddings found${NC}"
            echo "This might be due to:"
            echo "1. Invalid HuggingFace token"
            echo "2. Audio too short for diarization"
            echo "3. WhisperX diarization issue"
        fi
        
        break
        
    elif [ "$STATUS" = "failed" ]; then
        echo -e "${RED}❌ Prediction failed${NC}"
        
        ERROR=$(echo $STATUS_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin).get('error', 'Unknown error'))")
        echo "Error: $ERROR"
        
        # Get logs if available
        LOGS=$(echo $STATUS_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin).get('logs', ''))")
        if [ ! -z "$LOGS" ]; then
            echo ""
            echo "Logs:"
            echo "$LOGS"
        fi
        
        exit 1
        
    elif [ "$STATUS" = "canceled" ]; then
        echo -e "${YELLOW}⚠️  Prediction canceled${NC}"
        exit 1
    fi
    
    # Show progress
    echo -n "."
    ATTEMPTS=$((ATTEMPTS + 1))
done

if [ $ATTEMPTS -eq $MAX_ATTEMPTS ]; then
    echo ""
    echo -e "${RED}❌ Timeout waiting for prediction${NC}"
    echo "Prediction ID: $PREDICTION_ID"
    echo "Check status at: https://replicate.com/p/$PREDICTION_ID"
    exit 1
fi