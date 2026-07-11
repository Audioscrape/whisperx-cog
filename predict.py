"""
Production WhisperX Cog for Audioscrape
Handles everything from 5-minute news briefs to 8-hour government meetings
"""

import os
import json
import base64
import whisperx
import torch
import gc
import numpy as np
from typing import Dict, Any, Optional, List
from cog import BasePredictor, Input, Path
import warnings
warnings.filterwarnings("ignore")  # Suppress PyAnnote warnings


class Predictor(BasePredictor):
    """WhisperX optimized for podcast transcription with speaker tracking."""
    
    # Memory thresholds
    # 8h: government meetings routinely run 4-9h (GH#155). ASR is windowed
    # (constant VRAM) and batch_size already clamps to 4 for >60min audio;
    # the length-sensitive part is diarization clustering, tested to ~8h.
    # Beyond that, chunked transcription (app-side fan-out) is the plan.
    MAX_AUDIO_LENGTH_SECONDS = 8 * 3600  # 8 hours max
    CHUNK_LENGTH_SECONDS = 30 * 60  # Process in 30-min chunks for long audio
    
    def setup(self):
        """Load models once using Cog's setup pattern."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if self.device == "cuda" else "float32"
        
        # Initialize WhisperX
        
        # Load Whisper model (reused across predictions)
        print("Loading Whisper large-v3...")
        # Set ASR options including temperature
        asr_options = {
            "beam_size": 5,
            "best_of": 5,
            "patience": 1,
            "length_penalty": 1,
            "temperatures": [0.0],  # Use deterministic temperature
            "compression_ratio_threshold": 2.4,
            "no_speech_threshold": 0.6,
            "condition_on_previous_text": True,
            "initial_prompt": None,
            "suppress_tokens": [-1],
            "suppress_numerals": False,
        }
        
        self.model = whisperx.load_model(
            "large-v3",
            self.device,
            compute_type=compute_type,
            download_root="/tmp/whisper-models",  # Cog-friendly cache location
            asr_options=asr_options
        )
        
        # Pre-load English alignment (most common)
        print("Loading alignment models...")
        self.align_models = {}
        try:
            align_model, align_metadata = whisperx.load_align_model(
                language_code="en",
                device=self.device
            )
            self.align_models["en"] = (align_model, align_metadata)
        except Exception as e:
            print(f"Warning: Could not load English alignment model: {e}")
        
        # Diarization pipeline (loaded on-demand)
        self.diarize_model = None

        # CAM++ second speaker embedding (3D-Speaker, 192-dim). Loaded once;
        # import + load inside try so a failure here can NEVER break
        # transcription — the CAM++ embeddings are strictly additive.
        self.cam_model = None
        try:
            from modelscope.pipelines import pipeline as ms_pipeline
            self.cam_model = ms_pipeline(
                task="speaker-verification",
                model="iic/speech_campplus_sv_zh_en_16k-common_advanced",
            )
            print("Loaded CAM++ speaker-embedding model (192-dim)")
        except Exception as e:
            print(f"Warning: CAM++ model not loaded; cam embeddings disabled: {e}")

        # Setup complete
    
    def predict(
        self,
        audio: Path = Input(
            description="Audio file (supports up to 4 hours)"
        ),
        min_speakers: Optional[int] = Input(
            description="Minimum number of speakers (None = auto-detect)",
            default=None,
            ge=1,
            le=20
        ),
        max_speakers: Optional[int] = Input(
            description="Maximum number of speakers (None = auto-detect)",
            default=None,
            ge=1,
            le=20
        ),
        language: Optional[str] = Input(
            description="Language code (e.g., 'en'). Leave empty for auto-detect",
            default=None
        ),
        huggingface_token: str = Input(
            description="HuggingFace token for speaker diarization (required)"
        ),
        batch_size: int = Input(
            description="Batch size for transcription (lower for long audio)",
            default=8,
            ge=1,
            le=32
        ),
        enable_diarization: bool = Input(
            description="Enable speaker diarization",
            default=True
        ),
        return_word_timestamps: bool = Input(
            description="Return word-level timestamps",
            default=True
        )
    ) -> Dict[str, Any]:
        """
        Transcribe audio with speaker identification and embeddings.
        Handles everything from 5-minute podcasts to 8-hour meetings.
        """
        print(f"Processing: {audio}")
        
        # Load and validate audio
        audio_array = whisperx.load_audio(str(audio))
        duration_seconds = len(audio_array) / 16000
        duration_minutes = duration_seconds / 60
        
        # Validate duration
        if duration_seconds > self.MAX_AUDIO_LENGTH_SECONDS:
            raise ValueError(f"Audio too long: {duration_minutes:.1f} minutes (max: {self.MAX_AUDIO_LENGTH_SECONDS // 60} minutes)")
        
        # Adjust batch size for long audio (memory optimization)
        if duration_minutes > 60:
            batch_size = min(batch_size, 4)
            # Using reduced batch size for long audio
        
        # Step 1: Transcribe
        # Transcribing
        result = self._transcribe_audio(audio_array, batch_size, language)
        detected_language = result.get("language", "en")
        # Detected language
        
        # Step 2: Align (if requested)
        if return_word_timestamps:
            # Aligning words
            result = self._align_words(result, audio_array, detected_language)
        
        # Step 3: Diarize and extract embeddings (if requested)
        speaker_embeddings = None
        if enable_diarization:
            # Diarizing speakers
            result, speaker_embeddings = self._diarize_and_embed(
                audio_array, 
                result,
                huggingface_token,
                min_speakers,
                max_speakers
            )
        
        # Step 4: Post-process and validate
        segments = self._postprocess_segments(result.get("segments", []))

        # Step 4b: CAM++ second embedding per speaker (additive). Uses the
        # already-decoded audio_array + the diarized segments — no re-download.
        cam_speaker_embeddings = {}
        if enable_diarization:
            cam_speaker_embeddings = self._cam_speaker_embeddings(audio_array, segments)

        # Count statistics
        num_segments = len(segments)
        speakers = set(s.get("speaker") for s in segments if "speaker" in s)
        num_speakers = len(speakers)
        
        # Step 5: Memory cleanup (important for Cog)
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Build output with both segments AND speaker_embeddings at top level
        output = {
            "segments": segments,
            "metadata": {
                "duration_seconds": duration_seconds,
                "language": detected_language,
                "num_segments": num_segments,
                "num_speakers": num_speakers,
                "speakers": sorted(list(speakers)) if speakers else []
            }
        }
        
        # Add embeddings directly to output
        if speaker_embeddings:
            output["speaker_embeddings"] = speaker_embeddings
            output["metadata"]["embeddings_dims"] = len(next(iter(speaker_embeddings.values())))

        # CAM++ embeddings, kept in a separate field so existing consumers
        # are byte-for-byte unaffected until they opt in to reading it.
        if cam_speaker_embeddings:
            output["cam_speaker_embeddings"] = cam_speaker_embeddings
            output["metadata"]["cam_embeddings_dims"] = len(next(iter(cam_speaker_embeddings.values())))

        print(f"Complete: {num_segments} segments, {num_speakers} speakers")
        return output
    
    def _transcribe_audio(self, audio_array: np.ndarray, batch_size: int, language: Optional[str]) -> Dict:
        """Transcribe audio with automatic language detection."""
        try:
            # WhisperX FasterWhisperPipeline.transcribe() parameters
            result = self.model.transcribe(
                audio_array,
                batch_size=batch_size,
                language=language,  # None for auto-detect
                task="transcribe",  # transcribe or translate
                chunk_size=30,  # chunk size for processing
                print_progress=False
            )
            return result
        except Exception as e:
            # Fallback: Try with smaller batch size
            if batch_size > 1:
                return self._transcribe_audio(audio_array, batch_size // 2, language)
            raise
    
    def _align_words(self, result: Dict, audio_array: np.ndarray, language: str) -> Dict:
        """Align words with audio for precise timestamps."""
        # Load alignment model if not cached
        if language not in self.align_models:
            try:
                align_model, align_metadata = whisperx.load_align_model(
                    language_code=language,
                    device=self.device
                )
                self.align_models[language] = (align_model, align_metadata)
            except Exception:
                # Return original result without alignment
                return result
        
        try:
            align_model, align_metadata = self.align_models[language]
            
            # Perform alignment
            result = whisperx.align(
                result["segments"],
                align_model,
                align_metadata,
                audio_array,
                self.device,
                return_char_alignments=False  # We don't need character level
            )
        except Exception:
            pass  # Return original result
        
        return result
    
    def _diarize_and_embed(
        self, 
        audio_array: np.ndarray,
        result: Dict,
        hf_token: str,
        min_speakers: Optional[int],
        max_speakers: Optional[int]
    ) -> tuple[Dict, Optional[Dict]]:
        """Perform speaker diarization and extract embeddings."""
        
        # Load diarization model if needed
        if self.diarize_model is None:
            # Loading diarization model
            try:
                # Import DiarizationPipeline from the diarize module
                from whisperx.diarize import DiarizationPipeline
                self.diarize_model = DiarizationPipeline(
                    use_auth_token=hf_token,
                    device=self.device
                )
            except Exception:
                return result, None
        
        # Smart speaker count detection
        if min_speakers is None and max_speakers is None:
            # Auto-detect: Most podcasts have 2-4 speakers
            print("🔍 Auto-detecting speaker count...")
            min_speakers = None  # Let PyAnnote decide
            max_speakers = None
        elif min_speakers and not max_speakers:
            # If only min specified, set reasonable max
            max_speakers = min(min_speakers + 4, 10)
        elif max_speakers and not min_speakers:
            # If only max specified, set min to 1
            min_speakers = 1
        
        # Diarize with embeddings
        try:
            # Diarizing
            
            # KEY: Request embeddings from WhisperX
            diarize_params = {}
            if min_speakers is not None:
                diarize_params["min_speakers"] = min_speakers
            if max_speakers is not None:
                diarize_params["max_speakers"] = max_speakers
            
            # Get diarization and embeddings - based on PR #1085
            diarize_output = self.diarize_model(
                audio_array,
                return_embeddings=True,  # This flag triggers embedding extraction
                **diarize_params
            )
            
            # Handle output format from WhisperX PR #1085
            if isinstance(diarize_output, tuple) and len(diarize_output) == 2:
                diarize_segments, raw_embeddings = diarize_output
                speaker_embeddings = self._process_embeddings(raw_embeddings)
            else:
                diarize_segments = diarize_output
                speaker_embeddings = None
            
            # Assign speakers to words
            result = whisperx.assign_word_speakers(diarize_segments, result)
            
            return result, speaker_embeddings
            
        except Exception:
            # Return original result without speakers
            return result, None
    
    def _cam_speaker_embeddings(self, audio_array, segments: list) -> Dict[str, list]:
        """One L2-normalized CAM++ (192-dim) embedding per speaker, from the
        in-memory 16k audio.

        Strictly additive: any failure returns {} (or skips a speaker) so it
        can never affect transcription or the existing pyannote embeddings.
        """
        if self.cam_model is None:
            return {}
        try:
            import tempfile
            import soundfile as sf
            SR = 16000
            # Gather each speaker's segment time ranges.
            by_speaker: Dict[str, list] = {}
            for s in segments:
                spk = s.get("speaker")
                st, en = s.get("start"), s.get("end")
                if spk is None or st is None or en is None:
                    continue
                by_speaker.setdefault(spk, []).append((float(st), float(en)))

            out: Dict[str, list] = {}
            for spk, ranges in by_speaker.items():
                # Up to ~20s of this speaker's longest (clearest) turns.
                chunks, total = [], 0.0
                for st, en in sorted(ranges, key=lambda r: r[1] - r[0], reverse=True):
                    if total >= 20.0:
                        break
                    a, b = int(max(0.0, st) * SR), int(en * SR)
                    if b > a:
                        chunks.append(audio_array[a:b])
                        total += (en - st)
                if not chunks:
                    continue
                wav = np.concatenate(chunks).astype("float32")
                if len(wav) < SR:  # <1s of audio is too little to fingerprint
                    continue
                with tempfile.NamedTemporaryFile(suffix=".wav") as tf:
                    sf.write(tf.name, wav, SR)
                    res = self.cam_model([tf.name, tf.name], output_emb=True)
                emb = np.asarray(res["embs"][0], dtype="float32")
                norm = float(np.linalg.norm(emb))
                if norm > 0:
                    emb = emb / norm
                out[spk] = emb.tolist()
            return out
        except Exception as e:
            print(f"Warning: CAM++ embeddings failed (skipping): {e}")
            return {}

    def _postprocess_segments(self, segments: list) -> list:
        """Clean and validate segments for output."""
        processed = []
        
        for segment in segments:
            # Ensure all values are JSON-serializable
            clean_segment = {}
            for key, value in segment.items():
                if isinstance(value, (np.ndarray, torch.Tensor)):
                    if isinstance(value, torch.Tensor):
                        value = value.cpu().numpy()
                    value = value.tolist()
                elif isinstance(value, (np.float32, np.float64)):
                    value = float(value)
                elif isinstance(value, (np.int32, np.int64)):
                    value = int(value)
                clean_segment[key] = value
            
            # Ensure required fields
            if "text" in clean_segment and clean_segment["text"].strip():
                processed.append(clean_segment)
        
        return processed
    
    def _process_embeddings(self, raw_embeddings) -> Optional[Dict]:
        """Process raw embeddings from diarization into clean format."""
        if raw_embeddings is None:
            return None
            
        embeddings_dict = {}
        
        if isinstance(raw_embeddings, dict):
            for speaker, embedding in raw_embeddings.items():
                if isinstance(embedding, list):
                    embeddings_dict[speaker] = embedding
                elif isinstance(embedding, (torch.Tensor, np.ndarray)):
                    if isinstance(embedding, torch.Tensor):
                        embedding = embedding.cpu().numpy()
                    # Normalize embedding
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    embeddings_dict[speaker] = embedding.tolist()
        
        return embeddings_dict if embeddings_dict else None