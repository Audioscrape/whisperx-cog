#!/bin/bash
# Deploy Audioscrape WhisperX Cog to Replicate

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "🚀 Deploying Audioscrape WhisperX Cog to Replicate..."
echo "============================================================"

# Check if cog is installed
if ! command -v cog &> /dev/null; then
    echo "❌ Cog is not installed."
    echo "Install it with: pip install cog"
    exit 1
fi

# Change to the script directory
cd "$SCRIPT_DIR"

# Check if we're in the right directory
if [ ! -f "cog.yaml" ]; then
    echo "❌ Error: cog.yaml not found in $SCRIPT_DIR"
    echo "Please ensure the script is in the whisperx-cog directory"
    exit 1
fi

# Check cog.yaml is valid YAML
echo ""
echo "🔍 Checking Cog configuration..."
python3 -c "import yaml; yaml.safe_load(open('cog.yaml'))" 2>/dev/null || {
    echo "❌ cog.yaml is not valid YAML. Please fix syntax errors."
    exit 1
}
echo "✅ cog.yaml is valid"

# Build locally first (optional but recommended for testing)
echo ""
echo "🔨 Attempting local build to verify..."
if cog build --progress=plain 2>/dev/null; then
    echo "✅ Local build successful!"
else
    echo "⚠️  Local build skipped (Docker not available or permission denied)"
    echo "    The build will happen on Replicate's servers instead"
fi

# Reminder about authentication
echo ""
echo "📦 Make sure you're logged in to Replicate..."
echo "If not logged in, run: cog login"
echo ""

# Push to Replicate
echo ""
echo "📤 Pushing to Replicate..."
echo "Target: r8.im/audioscrape/whisperx"

cog push r8.im/audioscrape/whisperx || {
    echo "❌ Push failed."
    echo ""
    echo "Possible issues:"
    echo "1. You may not have permissions for the audioscrape organization"
    echo "2. The model may not exist yet on Replicate"
    echo ""
    echo "To create the model first:"
    echo "1. Go to https://replicate.com"
    echo "2. Create a new model named 'whisperx' under 'audioscrape' org"
    echo "3. Run this script again"
    exit 1
}

echo ""
echo "✅ Successfully deployed to Replicate!"
echo ""
echo "🔗 Model URL: https://replicate.com/audioscrape/whisperx"
echo ""
echo "📝 Next steps:"
echo "1. Test the model on Replicate website"
echo "2. Test via API with test_replicate.sh"
echo "3. Update compute_platform_client.rs to use 'audioscrape/whisperx'"
echo "4. Deploy Audioscrape to staging and test end-to-end"
echo ""
echo "🧪 Test command:"
echo "replicate run audioscrape/whisperx \\"
echo "  audio=@test_audio.mp3 \\"
echo "  huggingface_token=\$HF_TOKEN \\"
echo "  enable_diarization=true"