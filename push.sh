#!/bin/bash
# Quick push script for Audioscrape WhisperX Cog

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "📤 Pushing Audioscrape WhisperX to Replicate..."
echo "============================================"

# Change to the script directory
cd "$SCRIPT_DIR"

# Check if cog.yaml exists
if [ ! -f "cog.yaml" ]; then
    echo "❌ Error: cog.yaml not found"
    exit 1
fi

echo ""
echo "🚀 Running: cog push r8.im/audioscrape/whisperx"
echo ""

# Push to Replicate
cog push r8.im/audioscrape/whisperx

echo ""
echo "✅ Push complete!"
echo ""
echo "🔗 Model URL: https://replicate.com/audioscrape/whisperx"
echo ""
echo "📝 Next steps:"
echo "1. Test on Replicate: https://replicate.com/audioscrape/whisperx"
echo "2. Run API test: ./test_replicate.sh"
echo "3. Update compute_platform_client.rs to use 'audioscrape/whisperx'"