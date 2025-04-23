from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import json
import random
import numpy as np
from pydub import AudioSegment
import librosa
import torch
from ..data_processing.audio_distorter import AudioDistorter
import logging
import re
import traceback
from .prompts import get_mixer_prompts

logger = logging.getLogger(__name__)

class MixerDataset(Dataset):
    def __init__(self, 
        processed_dir: str,
        tokenizer: Any,
        config: Dict[str, Any],
        segment_duration: float = 5.0,
        max_seq_len: int = 512,  # Increased to handle larger JSON inputs
        sample_rate: int = 48000,  # Use 48000 Hz to match CLAP model
        num_distortions: int = 2,
        train_mode: bool = True,
        max_segments_per_song: int = 3,  # Max number of segments to include per song
        segment_selection_strategy: str = "sequential",  # "sequential", "random", "all"
        include_json: bool = True):  # Whether to include JSON metadata as text
        """
        Dataset for mixer training with stem-level distortions and song structure awareness.
        The model learns to generate mixing instructions for all stems and to combine segments.
        
        Args:
            processed_dir: Directory containing data (artist/song folders)
            tokenizer: Tokenizer for generating text prompts
            config: Configuration dictionary
            segment_duration: Duration of audio segments in seconds
            max_seq_len: Maximum sequence length for tokenized outputs
            sample_rate: Target sample rate for audio (48000 Hz for CLAP model)
            num_distortions: Number of distortions to apply to each stem
            train_mode: Whether to use training mode (includes distortions)
            max_segments_per_song: Maximum number of segments to include per song
            segment_selection_strategy: How to select segments from a song
            include_json: Whether to include JSON metadata in the prompt
        """
        self.processed_dir = processed_dir
        self.tokenizer = tokenizer
        self.config = config
        self.segment_duration = segment_duration
        self.max_seq_len = max_seq_len
        self.sample_rate = sample_rate
        self.num_distortions = num_distortions
        self.train_mode = train_mode
        self.max_segments_per_song = max_segments_per_song
        self.segment_selection_strategy = segment_selection_strategy
        self.include_json = include_json
        
        # Calculate fixed length for audio samples based on segment duration and sample rate
        self.fixed_audio_length = int(self.segment_duration * self.sample_rate)
        
        # Stem types to process
        self.stem_types = ["vocals", "drums", "bass", "other"]
        
        # Instructions for different mixing operations
        self.instructions_templates = {
            "change_volume": "Adjust the volume by {db} dB.",
            "pan": "Pan the audio to the {direction} by {value}.",
            "apply_low_pass": "Apply a low pass filter with cutoff at {cutoff} Hz.",
            "apply_high_pass": "Apply a high pass filter with cutoff at {cutoff} Hz.",
            "apply_normalize": "Normalize the audio levels.",
            "apply_delay": "Add a delay effect with {delay_ms} ms delay and {decay} decay.",
            "apply_reverb": "Add some reverb to the audio.",
            "apply_compression": "Apply compression with threshold at {threshold_db} dB and ratio {ratio}.",
            "apply_eq": "Adjust the EQ with low band: {low_gain} dB, mid band: {mid_gain} dB, high band: {high_gain} dB.",
            "apply_trim": "Trim {start_ms} ms from the start and {end_ms} ms from the end."
        }
        
        # Get enhanced prompts with character descriptions and output specifications
        self.mixer_prompts = get_mixer_prompts(30)  # Get all 30 mixer prompts
        
        # Keep the old prompt templates as fallback
        self.prompt_templates = [
            # Specific instruction prompts for individual stems
            "Fix the {stem} track with these adjustments: {instructions}",
            "Process the {stem} stem with these operations: {instructions}",
            "The {stem} track needs these modifications: {instructions}",
            
            # General prompts for overall song mixing
            "Mix these stems according to the provided song structure.",
            "Create a balanced mix following the song segments described.",
            "Combine these stems into a cohesive mix based on the provided structure.",
            "Process these stems into a final mix according to the song structure.",
            "Arrange and mix these stems following the provided segment information."
        ]
        
        # Always use on-the-fly chunking
        logger.info(f"Setting up on-the-fly chunking from {processed_dir}")
        self.songs = self._scan_data_directory(processed_dir)
        logger.info(f"Found {len(self.songs)} songs with segments and stems")
    
    def _scan_data_directory(self, data_dir: str) -> List[Dict]:
        """
        Scan the data directory to identify songs and their segments for on-the-fly chunking.
        
        Args:
            data_dir: Root directory containing artist folders
            
        Returns:
            List of dictionaries containing song and segment information
        """
        songs = []
        
        # Find all available artists
        artist_dirs = []
        for item in os.listdir(data_dir):
            full_path = os.path.join(data_dir, item)
            if os.path.isdir(full_path) and not item.startswith('.'):
                artist_dirs.append(full_path)
        
        logger.info(f"Found {len(artist_dirs)} artists in {data_dir}")
        
        # Find songs for each artist
        for artist_dir in artist_dirs:
            artist_name = os.path.basename(artist_dir)
            for song_dir_name in os.listdir(artist_dir):
                song_dir = os.path.join(artist_dir, song_dir_name)
                
                # Skip if not a directory
                if not os.path.isdir(song_dir):
                    continue
                
                # Find main song WAV and JSON files
                wav_file = None
                json_file = None
                for file in os.listdir(song_dir):
                    file_path = os.path.join(song_dir, file)
                    if file.endswith(".wav") and not os.path.isdir(file_path):
                        wav_file = file_path
                    elif file.endswith(".json"):
                        json_file = file_path
                
                # Skip if both WAV and JSON are not found
                if not wav_file or not json_file:
                    logger.warning(f"Skipping {song_dir}: WAV or JSON file not found")
                    continue
                
                # Find stems
                stems_dir = os.path.join(song_dir, "stems")
                stem_files = {}
                if os.path.exists(stems_dir) and os.path.isdir(stems_dir):
                    for stem_file in os.listdir(stems_dir):
                        if stem_file.endswith(".wav") and not os.path.isdir(os.path.join(stems_dir, stem_file)):
                            stem_name = os.path.splitext(stem_file)[0].lower()
                            stem_files[stem_name] = os.path.join(stems_dir, stem_file)
                
                # Verify we have all required stems
                missing_stems = [stem for stem in self.stem_types if stem not in stem_files]
                if missing_stems:
                    logger.warning(f"Skipping {song_dir}: Missing stems: {missing_stems}")
                    continue
                
                # Load segments from JSON
                try:
                    with open(json_file, 'r') as f:
                        json_data = json.load(f)
                    
                    # Get segments
                    segments = json_data.get("segments", [])
                    if not segments:
                        logger.warning(f"No segments found in JSON for {song_dir_name}")
                        continue
                    
                    # Skip songs without BPM metadata
                    if "bpm" not in json_data or json_data["bpm"] is None:
                        logger.warning(f"Skipping {song_dir_name}: BPM information missing")
                        continue
                    
                    # Create song entry
                    songs.append({
                        "artist": artist_name,
                        "song_name": song_dir_name,
                        "full_wav": wav_file,
                        "json_file": json_file,
                        "json_data": json_data,
                        "stem_files": stem_files,
                        "segments": segments,
                        "song_dir": song_dir
                    })
                    
                except Exception as e:
                    logger.error(f"Error loading JSON file {json_file}: {str(e)}")
                    continue
        
        return songs
    
    def _format_instructions(self, actions: List[Dict]) -> List[str]:
        """Format the actions into natural language instructions.
        
        Args:
            actions: List of action dictionaries that were applied
            
        Returns:
            List of formatted instruction strings
        """
        instructions = []
        
        for action in actions:
            tool = action["tool"]
            params = action["params"]
            
            if tool in self.instructions_templates:
                template = self.instructions_templates[tool]
                
                # Format based on specific tool params
                if tool == "change_volume":
                    db = params.get("db", 0)
                    instruction = template.format(db=db)
                
                elif tool == "pan":
                    pan_value = params.get("pan_value", 0)
                    direction = "right" if pan_value > 0 else "left"
                    value = abs(pan_value)
                    instruction = template.format(direction=direction, value=f"{value:.2f}")
                
                elif tool == "apply_low_pass" or tool == "apply_high_pass":
                    cutoff = params.get("cutoff", 1000)
                    instruction = template.format(cutoff=cutoff)
                
                elif tool == "apply_normalize":
                    instruction = template
                
                elif tool == "apply_delay":
                    delay_ms = params.get("delay_ms", 100)
                    decay = params.get("decay", 0.5)
                    instruction = template.format(delay_ms=delay_ms, decay=f"{decay:.2f}")
                
                elif tool == "apply_reverb":
                    instruction = template
                
                elif tool == "apply_compression":
                    threshold_db = params.get("threshold_db", -20)
                    ratio = params.get("ratio", 4.0)
                    instruction = template.format(threshold_db=threshold_db, ratio=f"{ratio:.1f}")
                
                elif tool == "apply_eq":
                    low_gain = params.get("low_gain", 0)
                    mid_gain = params.get("mid_gain", 0)
                    high_gain = params.get("high_gain", 0)
                    instruction = template.format(
                        low_gain=f"{low_gain:.1f}", 
                        mid_gain=f"{mid_gain:.1f}", 
                        high_gain=f"{high_gain:.1f}"
                    )
                    
                elif tool == "apply_trim":
                    start_ms = params.get("start_ms", 0)
                    end_ms = params.get("end_ms", 0)
                    instruction = template.format(start_ms=start_ms, end_ms=end_ms)
                    
                else:
                    # Default formatting
                    instruction = template
                    
                instructions.append(instruction)
        
        return instructions
    
    def _generate_tool_tokens(self, actions: List[Dict], stem_name: Optional[str] = None, loop_count: int = 1) -> str:
        """Generate tool tokens for model training from actions.
        
        This method will generate tokens for ALL possible distortions, with zeroed
        parameters for distortions that weren't applied. This ensures the model
        always learns to make predictions for the complete set of tools.
        
        Args:
            actions: List of action dictionaries that were applied
            stem_name: Name of the stem (for stem-specific token generation)
            loop_count: Loop count for rhythmic elements
            
        Returns:
            String of token representations
        """
        tokens = []
        
        # Add stem specifier if provided
        if stem_name:
            tokens.append(f"<{stem_name}>")
        
        # Add loop count for rhythmic stems
        if stem_name and any(x in stem_name for x in ["bass", "drums"]) and loop_count > 1:
            tokens.append(f"<loop:{loop_count}>")
        elif stem_name and loop_count == 0:  # Specific token for empty chunks
            tokens.append("<empty>")
            # For empty chunks, we only need to add the stem and empty token
            return " ".join(tokens)
        
        # Define the complete set of distortions and their default parameters with compact token format
        all_distortions = {
            "change_volume": {"db": 0.0},
            "pan": {"pan_value": 0.0},
            "apply_low_pass": {"cutoff": 20000},  # No effect at 20kHz
            "apply_high_pass": {"cutoff": 20},    # No effect at 20Hz
            "apply_normalize": {},
            "apply_delay": {"delay_ms": 0, "decay": 0.0},
            "apply_reverb": {},
            "apply_compression": {"threshold_db": -30, "ratio": 1.0},  # 1.0 ratio = no compression
            "apply_eq": {"low_gain": 0.0, "mid_gain": 0.0, "high_gain": 0.0},
            "apply_trim": {"start_ms": 0, "end_ms": 0},
            "loop_section": {"start_ms": 0, "end_ms": 0, "count": 1},
            "duplicate_and_shift": {"offset_ms": 0},
            "speed_change": {"factor": 1.0}  # 1.0 = no change
        }
        
        # Convert actions to a dictionary for easy lookup
        applied_actions = {}
        for action in actions:
            tool = action["tool"]
            # Map from AudioDistorter names to mixer tool names
            if tool == "low_pass":
                tool = "apply_low_pass"
            elif tool == "high_pass":
                tool = "apply_high_pass"
            elif tool == "normalize":
                tool = "apply_normalize"
            elif tool == "delay":
                tool = "apply_delay"
            elif tool == "reverb":
                tool = "apply_reverb"
            elif tool == "compression":
                tool = "apply_compression"
            elif tool == "eq":
                tool = "apply_eq"
            elif tool == "trim":
                tool = "apply_trim"
            
            applied_actions[tool] = action["params"]
        
        # Use more compact token format for tools
        tool_token_map = {
            "change_volume": "vol",
            "pan": "pan",
            "apply_low_pass": "lpf",
            "apply_high_pass": "hpf",
            "apply_normalize": "norm",
            "apply_delay": "dly",
            "apply_reverb": "rvb",
            "apply_compression": "cmp",
            "apply_eq": "eq",
            "apply_trim": "trim",
            "loop_section": "loop",
            "duplicate_and_shift": "dupe",
            "speed_change": "spd"
        }
        
        # Process all possible distortions
        for tool, default_params in all_distortions.items():
            # Skip speed_change and loop_section for vocal stems unless specifically applied
            if stem_name and "vocals" in stem_name:
                if tool in ["loop_section", "speed_change"] and tool not in applied_actions:
                    continue
            
            # Skip tools that have no effect with default parameters
            # Only add tokens for tools that are applied or that modify the audio
            if tool not in applied_actions and tool not in ["apply_normalize", "apply_reverb"]:
                # Check if this tool with default params would do anything
                has_effect = False
                
                if tool == "change_volume" and abs(default_params.get("db", 0)) > 0.1:
                    has_effect = True
                elif tool == "pan" and abs(default_params.get("pan_value", 0)) > 0.05:
                    has_effect = True
                elif tool == "apply_low_pass" and default_params.get("cutoff", 20000) < 19000:
                    has_effect = True
                elif tool == "apply_high_pass" and default_params.get("cutoff", 20) > 30:
                    has_effect = True
                elif tool == "apply_delay" and (default_params.get("delay_ms", 0) > 5 and default_params.get("decay", 0) > 0.05):
                    has_effect = True
                elif tool == "apply_compression" and (default_params.get("ratio", 1.0) > 1.1):
                    has_effect = True
                elif tool == "apply_eq" and (abs(default_params.get("low_gain", 0)) > 0.1 or 
                                            abs(default_params.get("mid_gain", 0)) > 0.1 or 
                                            abs(default_params.get("high_gain", 0)) > 0.1):
                    has_effect = True
                elif tool == "apply_trim" and (default_params.get("start_ms", 0) > 0 or default_params.get("end_ms", 0) > 0):
                    has_effect = True
                elif tool == "loop_section" and default_params.get("count", 1) > 1:
                    has_effect = True
                elif tool == "duplicate_and_shift" and default_params.get("offset_ms", 0) > 0:
                    has_effect = True
                elif tool == "speed_change" and abs(default_params.get("factor", 1.0) - 1.0) > 0.01:
                    has_effect = True
                
                if not has_effect:
                    continue  # Skip this tool as it has no effect with default parameters
            
            # Get parameters (either applied or default)
            if tool in applied_actions:
                params = applied_actions[tool]
                
                # Apply inverse transformations for some parameters
                if tool == "change_volume" and "db" in params:
                    params["db"] = -params["db"]  # Reverse volume change
                elif tool == "pan" and "pan_value" in params:
                    params["pan_value"] = -params["pan_value"]  # Reverse pan
                elif tool == "apply_eq":
                    # Reverse EQ gain values
                    for key in ["low_gain", "mid_gain", "high_gain"]:
                        if key in params:
                            params[key] = -params[key]
            else:
                params = default_params
            
            # Use compact token name for the tool
            token_name = tool_token_map.get(tool, tool)
            
            # Format the parameters in a compact way
            if not params:  # No parameters
                tokens.append(f"<{token_name}>")
            elif tool == "change_volume":
                # Volume: only include if non-zero
                db = params.get("db", 0)
                if abs(db) > 0.1:  # Only include if significant
                    tokens.append(f"<{token_name}:{db:.1f}>")
            elif tool == "pan":
                # Pan: only include if non-zero
                pan_value = params.get("pan_value", 0)
                if abs(pan_value) > 0.05:  # Only include if significant
                    tokens.append(f"<{token_name}:{pan_value:.1f}>")
            elif tool == "apply_low_pass" or tool == "apply_high_pass":
                # Filters: only include if they have a significant effect
                cutoff = params.get("cutoff", 20000 if tool == "apply_low_pass" else 20)
                if (tool == "apply_low_pass" and cutoff < 19000) or (tool == "apply_high_pass" and cutoff > 30):
                    tokens.append(f"<{token_name}:{cutoff}>")
            elif tool == "apply_delay":
                # Delay: only include if delay_ms and decay are non-trivial
                delay_ms = params.get("delay_ms", 0)
                decay = params.get("decay", 0)
                if delay_ms > 5 and decay > 0.05:
                    tokens.append(f"<{token_name}:{delay_ms},{decay:.1f}>")
            elif tool == "apply_compression":
                # Compression: only include if ratio is significant
                threshold_db = params.get("threshold_db", -30)
                ratio = params.get("ratio", 1.0)
                if ratio > 1.1:
                    tokens.append(f"<{token_name}:{threshold_db},{ratio:.1f}>")
            elif tool == "apply_eq":
                # EQ: only include gains that are non-zero
                low_gain = params.get("low_gain", 0)
                mid_gain = params.get("mid_gain", 0)
                high_gain = params.get("high_gain", 0)
                
                # Only include if at least one gain is significant
                if abs(low_gain) > 0.1 or abs(mid_gain) > 0.1 or abs(high_gain) > 0.1:
                    tokens.append(f"<{token_name}:{low_gain:.1f},{mid_gain:.1f},{high_gain:.1f}>")
            elif tool == "apply_trim":
                # Trim: only include if non-zero
                start_ms = params.get("start_ms", 0)
                end_ms = params.get("end_ms", 0)
                if start_ms > 0 or end_ms > 0:
                    tokens.append(f"<{token_name}:{start_ms},{end_ms}>")
            elif tool == "loop_section":
                # Loop section: only include if count > 1
                count = params.get("count", 1)
                if count > 1:
                    start_ms = params.get("start_ms", 0)
                    end_ms = params.get("end_ms", 0)
                    tokens.append(f"<{token_name}:{start_ms},{end_ms},{count}>")
            elif tool == "speed_change":
                # Speed change: only include if factor != 1.0
                factor = params.get("factor", 1.0)
                if abs(factor - 1.0) > 0.01:
                    tokens.append(f"<{token_name}:{factor:.2f}>")
            elif tool == "duplicate_and_shift":
                # Duplicate and shift: only include if offset is non-zero
                offset_ms = params.get("offset_ms", 0)
                if offset_ms > 0:
                    tokens.append(f"<{token_name}:{offset_ms}>")
            else:
                # Simple tools with no parameters
                tokens.append(f"<{token_name}>")
        
        return " ".join(tokens)
    
    def _format_song_structure(self, segments):
        """Format song structure information from segments data."""
        structure = []
        
        # Get BPM from current song data if available
        bpm = 120  # Default BPM
        if hasattr(self, "current_song_bpm"):
            bpm = self.current_song_bpm
        
        # Calculate eight count duration in seconds
        eight_count_duration_sec = (60.0 / bpm) * 8
        
        for i, segment in enumerate(segments):
            start = segment["start"]
            end = segment["end"]
            label = segment["label"]
            duration = end - start
            
            # Convert to eight counts
            start_eight_count = start / eight_count_duration_sec
            end_eight_count = end / eight_count_duration_sec
            duration_eight_counts = duration / eight_count_duration_sec
            
            # Format with both time measures
            structure.append(
                f"Segment {i+1}: {label}, "
                f"start={start:.2f}s ({start_eight_count:.2f} eight counts), "
                f"end={end:.2f}s ({end_eight_count:.2f} eight counts), "
                f"duration={duration:.2f}s ({duration_eight_counts:.2f} eight counts)"
            )
        
        return "Song Structure:\n" + "\n".join(structure) + "\nNote: One eight count equals 8 beats (4 seconds at 120 BPM)"
    
    def _calculate_eight_count_duration(self, bpm):
        """
        Calculate the duration of an eight count (8 beats) in milliseconds.
        
        Args:
            bpm: Beats per minute
            
        Returns:
            Duration of eight count in milliseconds
        """
        if not bpm or bpm < 20 or bpm > 300:  # Validate BPM is in a reasonable range
            bpm = 120  # Default to 120 BPM if invalid
        
        # Calculate duration of one beat in milliseconds
        beat_duration_ms = (60.0 / bpm) * 1000
        
        # 8 beats make an eight count
        eight_count_ms = beat_duration_ms * 8
        
        return eight_count_ms
    
    def _seconds_to_eight_counts(self, seconds, bpm=120):
        """
        Convert time in seconds to musical eight counts.
        
        Args:
            seconds: Time in seconds
            bpm: Beats per minute (default: 120)
            
        Returns:
            Time in eight counts (where one eight count = 8 beats)
        """
        # Ensure BPM is not None
        if bpm is None:
            bpm = 120.0  # Use default if None
            
        # Calculate duration of one beat in seconds
        beat_duration_sec = 60.0 / bpm
        
        # Calculate duration of one eight count (8 beats) in seconds
        eight_count_duration_sec = beat_duration_sec * 8
        
        # Convert seconds to eight counts
        eight_counts = seconds / eight_count_duration_sec
        
        return eight_counts
    
    def _extract_segment_audio(self, audio_file: str, segment: Dict, trim_silence: bool = False, stem_name: Optional[str] = None) -> Tuple[List[Union[AudioSegment, np.ndarray]], List[int], List[Dict]]:
        """Extract a segment from a full audio file with beat-based chunking.
        
        Args:
            audio_file: Path to the audio file
            segment: Segment dictionary with start and end times
            trim_silence: Whether to trim silence from the segment
            stem_name: Name of the stem (used for metadata tagging)
            
        Returns:
            Tuple containing:
                - List of audio segment chunks (AudioSegment or numpy array)
                - List of loop counts for each chunk
                - List of metadata dictionaries for each chunk
        """
        full_audio = AudioSegment.from_wav(audio_file)
        
        # Convert segment times to milliseconds
        start_time = segment["start"] * 1000
        end_time = segment["end"] * 1000
        
        # Extract the segment
        segment_audio = full_audio[start_time:end_time]
        
        # Check for silence or very low volume in the entire segment
        is_silent = segment_audio.dBFS < -40
        
        # For silent non-vocal stems, create a special silent segment with metadata indicating silence
        if is_silent and stem_name != "vocals":
            # Create a silent AudioSegment of the proper length
            silent_duration = int(self.segment_duration * 1000)  # Convert to ms
            silent_segment = AudioSegment.silent(duration=silent_duration, frame_rate=self.sample_rate)
            
            # Create metadata that indicates this is a silent chunk
            silent_metadata = {
                "stem_type": stem_name,
                "segment_label": segment.get("label", "unknown"),
                "position": 0,
                "is_silent": True,
                "loop_count": 0,  # Special loop count for silent chunks
                "start_sec": segment["start"],
                "end_sec": segment["end"],
                "start_eight_count": 0,
                "end_eight_count": 0
            }
            
            # Return the silent segment with a loop count of 0 and special metadata
            return [silent_segment], [0], [silent_metadata]
        
        # Get the BPM for beat-based chunking - we can safely use it since we filter out songs without BPM
        bpm = self.current_song_bpm
        
        # Ensure BPM is not None to avoid division errors
        if bpm is None:
            bpm = 120.0  # Use default 120 BPM if not available
        
        # Calculate chunk sizes based on musical bars
        # 8 beats = 2 bars in 4/4 time
        eight_beat_ms = (60.0 / bpm) * 8 * 1000
        four_beat_ms = (60.0 / bpm) * 4 * 1000
        
        # Prefer chunking by 8 beats (2 bars), but fall back to 4 beats (1 bar) if needed
        # or even smaller chunks if the segment is very short
        segment_duration_ms = segment_audio.duration_seconds * 1000
        
        # Get beat information from the current song data
        beats = []
        downbeats = []
        bpm = 120  # Default BPM
        beat_positions_ms = []
        
        # Try to get beat information from the song JSON data
        if hasattr(self, "current_song_json"):
            json_data = self.current_song_json
            
            # Get BPM
            if "bpm" in json_data:
                bpm = json_data["bpm"]
            
            # Get beat positions if available
            if "beats" in json_data:
                beats = json_data["beats"]
            
            # Get downbeat positions if available
            if "downbeats" in json_data:
                downbeats = json_data["downbeats"]
        
        # Calculate beat positions
        if downbeats:
            # Convert downbeat times to milliseconds relative to segment start
            for beat_time in downbeats:
                beat_ms = beat_time * 1000
                if start_time <= beat_ms < end_time:
                    beat_positions_ms.append(beat_ms - start_time)
        elif beats:
            # If no downbeats, use regular beats
            for beat_time in beats:
                beat_ms = beat_time * 1000
                if start_time <= beat_ms < end_time:
                    beat_positions_ms.append(beat_ms - start_time)
        
        # Sort beat positions
        beat_positions_ms.sort()
        
        # Reference BPM and eight beat count duration (always 4 seconds at 120 BPM)
        reference_bpm = 120
        
        # Check if BPM is None and provide a default
        if bpm is None:
            bpm = reference_bpm  # Use 120 BPM as default
            
        # Calculate eight beat count in milliseconds at the reference BPM
        reference_eight_beat_ms = (60.0 / reference_bpm) * 8 * 1000  # 4000ms
        
        # Calculate eight beat count duration for the song's actual BPM
        eight_beat_ms = (60.0 / bpm) * 8 * 1000
        
        # Calculate speed adjustment factor to normalize to reference BPM
        speed_factor = bpm / reference_bpm
        
        # Get segment length in milliseconds
        segment_length_ms = len(segment_audio)
        
        # If segment is too short for a full eight beat count, use what we have
        if segment_length_ms < eight_beat_ms:
            # Normalize speed to reference BPM
            original_frame_rate = segment_audio.frame_rate
            new_frame_rate = int(original_frame_rate * speed_factor)
            
            normalized_audio = segment_audio._spawn(
                segment_audio.raw_data,
                overrides={"frame_rate": new_frame_rate}
            )
            normalized_audio = normalized_audio.set_frame_rate(original_frame_rate)
            
            # If it's still too short after normalization, pad with silence
            if len(normalized_audio) < reference_eight_beat_ms:
                silence_needed = reference_eight_beat_ms - len(normalized_audio)
                padding = AudioSegment.silent(duration=silence_needed, frame_rate=original_frame_rate)
                normalized_audio = normalized_audio + padding
            
            # Ensure exact length by trimming if needed
            normalized_audio = normalized_audio[:int(reference_eight_beat_ms)]
            
            # Check if this short segment is mostly silent
            if normalized_audio.dBFS < -30:
                # Use a loop count of 0 to indicate an empty/silent chunk
                loop_count = 0
            else:
                # For short segments with audio, we'll use a loop count of 1
                loop_count = 1
                
                # For rhythmic stems, increase loop count based on segment length
                if stem_name in ["bass", "drums"]:
                    # Calculate number of times this would need to loop to cover remaining segment
                    potential_loops = max(1, int(segment_length_ms / len(normalized_audio)))
                    loop_count = min(8, potential_loops)  # Cap at 8 loops
            
            metadata = {
                "stem_type": stem_name,
                "segment_label": segment.get("label", "unknown"),
                "position": 0,
                "bpm": bpm,
                "speed_factor": speed_factor,
                "loop_count": loop_count,
                "start_sec": segment["start"],
                "end_sec": segment["end"],
                "start_eight_count": self._seconds_to_eight_counts(segment["start"], bpm),
                "end_eight_count": self._seconds_to_eight_counts(segment["end"], bpm)
            }
            
            return [normalized_audio], [loop_count], [metadata]
        
        # Find chunk starting points
        chunk_start_points = [0]  # Always include start of segment
        
        # Add beat-aligned starting points if available
        if beat_positions_ms:
            # Include the first beat position if it's not too close to the start
            if beat_positions_ms[0] > 100:
                chunk_start_points.append(int(beat_positions_ms[0]))
            
            # Add more starting points at eight beat intervals if segment is long enough
            for beat_pos in beat_positions_ms:
                if beat_pos >= eight_beat_ms:  # Only use beats that are at least one eight beat count in
                    chunk_start_points.append(int(beat_pos))
        
        # Deduplicate and sort starting points
        chunk_start_points = sorted(list(set(chunk_start_points)))
        
        # Limit number of chunks to avoid excessive memory use
        max_chunks = 6  # Maximum chunks per segment
        if len(chunk_start_points) > max_chunks:
            # Keep the first chunk and evenly sample the rest
            first_point = chunk_start_points[0]
            remaining_points = chunk_start_points[1:]
            if len(remaining_points) > max_chunks - 1:
                # Sample evenly from remaining points
                step = len(remaining_points) // (max_chunks - 1)
                sampled_points = [remaining_points[i * step] for i in range(max_chunks - 1)]
                chunk_start_points = [first_point] + sampled_points
        
        # Extract chunks
        chunks = []
        metadata_list = []
        loop_counts = []
        
        for i, start_pos in enumerate(chunk_start_points):
            # Don't extract if we'd go beyond the segment
            if start_pos + eight_beat_ms > segment_length_ms:
                continue
                
            # Extract exactly eight beats
            chunk = segment_audio[start_pos:int(start_pos + eight_beat_ms)]
            
            # Check if this chunk is silent
            if chunk.dBFS < -40:
                # Skip silent chunks
                continue
                
            # Normalize speed to reference BPM
            original_frame_rate = chunk.frame_rate
            new_frame_rate = int(original_frame_rate * speed_factor)
            
            normalized_chunk = chunk._spawn(
                chunk.raw_data,
                overrides={"frame_rate": new_frame_rate}
            )
            normalized_chunk = normalized_chunk.set_frame_rate(original_frame_rate)
            
            # Ensure exact length by trimming if needed
            normalized_chunk = normalized_chunk[:int(reference_eight_beat_ms)]
            
            # Skip if the chunk is too short after normalization
            if len(normalized_chunk) < reference_eight_beat_ms * 0.9:
                continue
                
            # For rhythmic stems, calculate loop count based on pattern repetition and segment length
            loop_count = 1
            if stem_name in ["bass", "drums"]:
                # Check if the chunk is highly repetitive
                if self._check_high_similarity(normalized_chunk):
                    # Repetitive patterns can loop more
                    remaining_duration = segment_length_ms - start_pos
                    potential_loops = max(1, int(remaining_duration / eight_beat_ms))
                    loop_count = min(8, potential_loops)  # Cap at 8 loops
                else:
                    # Calculate total segment coverage
                    total_segment = end_time - start_time
                    chunk_duration = eight_beat_ms
                    coverage_ratio = chunk_duration / total_segment
                    
                    # If chunk covers a significant portion, don't loop as much
                    if coverage_ratio > 0.5:
                        loop_count = 1
                    else:
                        # Calculate number of times this would need to loop to cover remaining segment
                        remaining_duration = segment_length_ms - start_pos
                        potential_loops = max(1, int(remaining_duration / eight_beat_ms))
                        loop_count = min(4, potential_loops)  # More conservative looping
            
            # For vocal stems, check if repeating this chunk makes sense
            elif stem_name == "vocals":
                # We generally don't want to loop vocals, but for short segments or
                # segments with consistent patterns (e.g., ad-libs), we might loop
                if self._check_high_similarity(normalized_chunk):
                    remaining_duration = segment_length_ms - start_pos
                    potential_loops = max(1, int(remaining_duration / eight_beat_ms))
                    loop_count = min(2, potential_loops)  # Very conservative looping for vocals
            
            chunks.append(normalized_chunk)
            loop_counts.append(loop_count)
            
            # Store metadata for this chunk
            metadata = {
                "stem_type": stem_name,
                "segment_label": segment.get("label", "unknown"),
                "position": i,
                "bpm": bpm,
                "speed_factor": speed_factor,
                "loop_count": loop_count,
                "start_sec": segment["start"],
                "end_sec": segment["end"],
                "start_eight_count": self._seconds_to_eight_counts(segment["start"], bpm),
                "end_eight_count": self._seconds_to_eight_counts(segment["end"], bpm)
            }
            metadata_list.append(metadata)
        
        # Handle case where no chunks were extracted (segment too short or no valid starting points)
        if not chunks:
            # Create one chunk at the beginning
            if segment_length_ms >= eight_beat_ms:
                chunk = segment_audio[:int(eight_beat_ms)]
            else:
                # If segment is shorter than eight beats, use the entire segment
                chunk = segment_audio
                # Pad with silence if needed
                if len(chunk) < eight_beat_ms:
                    silence_needed = eight_beat_ms - len(chunk)
                    padding = AudioSegment.silent(duration=silence_needed, frame_rate=chunk.frame_rate)
                    chunk = chunk + padding
            
            # Check if this chunk is silent
            is_silent = chunk.dBFS < -40
            loop_count = 0 if is_silent else 1  # Use 0 to indicate silence
            
            # Normalize speed to reference BPM
            original_frame_rate = chunk.frame_rate
            new_frame_rate = int(original_frame_rate * speed_factor)
            
            normalized_chunk = chunk._spawn(
                chunk.raw_data,
                overrides={"frame_rate": new_frame_rate}
            )
            normalized_chunk = normalized_chunk.set_frame_rate(original_frame_rate)
            
            # Ensure exact length by trimming if needed
            normalized_chunk = normalized_chunk[:int(reference_eight_beat_ms)]
            
            chunks = [normalized_chunk]
            loop_counts = [loop_count]
            
            # Store metadata for this chunk
            metadata = {
                "stem_type": stem_name,
                "segment_label": segment.get("label", "unknown"),
                "position": 0,
                "bpm": bpm,
                "speed_factor": speed_factor,
                "loop_count": loop_count,
                "start_sec": segment["start"],
                "end_sec": segment["end"],
                "start_eight_count": self._seconds_to_eight_counts(segment["start"], bpm),
                "end_eight_count": self._seconds_to_eight_counts(segment["end"], bpm),
                "is_empty": is_silent
            }
            metadata_list = [metadata]
        
        # Check for duplicative chunks (highly similar audio content)
        unique_chunks = []
        unique_loop_counts = []
        unique_metadata = []
        
        for i, chunk in enumerate(chunks):
            # Check for silent/empty chunks
            is_silent = chunk.dBFS < -40
            loop_count = loop_counts[i]
            
            # For silent chunks, only keep one and set loop_count to 0
            if is_silent:
                # If we already have a silent chunk, skip this one
                if any(m.get("is_empty", False) for m in unique_metadata):
                    continue
                
                # Otherwise, keep this silent chunk with loop_count 0
                unique_chunks.append(chunk)
                unique_loop_counts.append(0)  # Special loop count for empty chunks
                
                # Update metadata to mark as empty
                metadata = metadata_list[i].copy()
                metadata["is_empty"] = True
                metadata["loop_count"] = 0
                unique_metadata.append(metadata)
                continue
            
            # Skip similar chunks except for vocal stems (we want to keep all vocal chunks)
            if stem_name != "vocals" and self._is_similar_to_existing_chunks(chunk, unique_chunks):
                # Find the chunk this is similar to and update its loop count
                for j, existing_chunk in enumerate(unique_chunks):
                    if self._audio_to_numpy(chunk).shape == self._audio_to_numpy(existing_chunk).shape:
                        try:
                            corr = np.corrcoef(
                                self._audio_to_numpy(chunk), 
                                self._audio_to_numpy(existing_chunk)
                            )[0, 1]
                            
                            if corr > 0.8:  # Similarity threshold
                                # Update the loop count to be the maximum
                                unique_loop_counts[j] = max(unique_loop_counts[j], loop_count)
                                # Update the metadata to reflect the higher loop count
                                if unique_loop_counts[j] > unique_metadata[j].get("loop_count", 1):
                                    unique_metadata[j]["loop_count"] = unique_loop_counts[j]
                                break
                        except:
                            # In case of numerical issues, continue
                            pass
                continue
            
            unique_chunks.append(chunk)
            unique_loop_counts.append(loop_count)
            unique_metadata.append(metadata_list[i])
        
        # For empty sets, add a silent chunk
        if not unique_chunks:
            silent_duration = self.segment_duration * 1000  # Convert to ms
            silent_chunk = AudioSegment.silent(duration=silent_duration, frame_rate=48000)
            
            unique_chunks = [silent_chunk]
            unique_loop_counts = [0]  # Special loop count for empty chunks
            
            metadata = {
                "stem_type": stem_name,
                "segment_label": segment.get("label", "unknown"),
                "position": 0,
                "bpm": bpm,
                "speed_factor": speed_factor,
                "loop_count": 0,
                "is_empty": True,
                "start_sec": segment["start"],
                "end_sec": segment["end"],
                "start_eight_count": self._seconds_to_eight_counts(segment["start"], bpm),
                "end_eight_count": self._seconds_to_eight_counts(segment["end"], bpm)
            }
            unique_metadata = [metadata]
        
        # Return the processed chunks with their loop counts and metadata
        return unique_chunks, unique_loop_counts, unique_metadata
    
    def _check_high_similarity(self, audio_segment: Union[AudioSegment, np.ndarray], threshold: float = 0.85) -> bool:
        """
        Check if an audio segment is highly self-similar (repetitive).
        
        Args:
            audio_segment: The audio segment to check (AudioSegment or numpy array)
            threshold: Similarity threshold (0-1)
            
        Returns:
            bool: True if highly similar
        """
        # Convert to numpy array for analysis
        samples = self._audio_to_numpy(audio_segment)
        
        # For very short segments, can't really analyze
        if len(samples) < 1000:
            return False
        
        # Split into two halves
        half_len = len(samples) // 2
        first_half = samples[:half_len]
        second_half = samples[half_len:2*half_len]
        
        # If they're different lengths, pad the shorter one
        if len(first_half) > len(second_half):
            padding = np.zeros(len(first_half) - len(second_half))
            second_half = np.concatenate([second_half, padding])
        elif len(second_half) > len(first_half):
            padding = np.zeros(len(second_half) - len(first_half))
            first_half = np.concatenate([first_half, padding])
        
        # Calculate correlation coefficient
        try:
            corr = np.corrcoef(first_half, second_half)[0, 1]
            return corr > threshold
        except:
            # In case of numerical issues
            return False
    
    def _is_similar_to_existing_chunks(self, new_chunk: Union[AudioSegment, np.ndarray], 
                                      existing_chunks: List[Union[AudioSegment, np.ndarray]], 
                                      threshold: float = 0.8) -> bool:
        """
        Check if a new chunk is similar to any of the existing chunks.
        
        Args:
            new_chunk: The new audio chunk to check (AudioSegment or numpy array)
            existing_chunks: List of existing audio chunks (AudioSegment or numpy array)
            threshold: Similarity threshold (0-1)
            
        Returns:
            bool: True if similar to any existing chunk
        """
        if not existing_chunks:
            return False
        
        # Check if the chunk is mostly silent
        new_samples = self._audio_to_numpy(new_chunk)
        rms = np.sqrt(np.mean(np.square(new_samples)))
        if rms < 0.001:  # Very low energy
            return True  # Consider silent chunks as similar to avoid duplicates
        
        for chunk in existing_chunks:
            existing_samples = self._audio_to_numpy(chunk)
            
            # Check if existing chunk is mostly silent
            existing_rms = np.sqrt(np.mean(np.square(existing_samples)))
            if existing_rms < 0.001:  # Very low energy
                continue  # Skip comparison with silent chunks
            
            # If they're significantly different in length, they're not similar
            if abs(len(new_samples) - len(existing_samples)) > len(new_samples) * 0.2:
                continue
            
            # Trim to the same length for comparison
            min_len = min(len(new_samples), len(existing_samples))
            new_trimmed = new_samples[:min_len]
            existing_trimmed = existing_samples[:min_len]
            
            # Calculate correlation
            try:
                corr = np.corrcoef(new_trimmed, existing_trimmed)[0, 1]
                if corr > threshold:
                    return True
            except:
                # In case of numerical issues
                continue
        
        return False
    
    def _audio_to_numpy(self, audio_segment: Union[AudioSegment, np.ndarray], target_sr: int = 48000) -> np.ndarray:
        """Convert pydub AudioSegment or numpy array to a normalized numpy array.
        
        Args:
            audio_segment: Audio segment to convert (either AudioSegment or numpy array)
            target_sr: Target sample rate for resampling
            
        Returns:
            Normalized audio as numpy array
        """
        # If already a numpy array, return it directly
        if isinstance(audio_segment, np.ndarray):
            return audio_segment
            
        # For AudioSegment objects, process normally
        try:
            # Get audio data as array of samples
            samples = np.array(audio_segment.get_array_of_samples())
            
            # Convert to float32 in range [-1, 1]
            if audio_segment.sample_width == 2:  # 16-bit samples
                samples = samples.astype(np.float32) / 32768.0
            elif audio_segment.sample_width == 3:  # 24-bit samples
                samples = samples.astype(np.float32) / 8388608.0
            elif audio_segment.sample_width == 4:  # 32-bit samples
                samples = samples.astype(np.float32) / 2147483648.0
            
            # Reshape for mono/stereo
            if audio_segment.channels == 2:
                samples = samples.reshape((-1, 2))
                # Average to mono if needed
                samples = samples.mean(axis=1)
            
            # Resample if needed
            if audio_segment.frame_rate != target_sr:
                samples = librosa.resample(
                    samples, 
                    orig_sr=audio_segment.frame_rate, 
                    target_sr=target_sr
                )
            
            return samples
        except AttributeError as e:
            # Log the error and provide more context
            logger.error(f"Invalid audio segment type: {type(audio_segment)}")
            raise AttributeError(f"Expected AudioSegment or numpy.ndarray, got {type(audio_segment)}") from e
    
    def _format_json_for_prompt(self, json_data):
        """Format JSON data for inclusion in the prompt."""
        # Extract the essential information
        segments = json_data.get("segments", [])
        bpm = json_data.get("bpm", "unknown")
        
        # Format for readability
        formatted = f"BPM: {bpm}\n\nSegments:\n"
        for i, segment in enumerate(segments):
            start = segment["start"]
            end = segment["end"]
            label = segment["label"]
            formatted += f"{i+1}. {label}: {start:.2f}s - {end:.2f}s\n"
        
        return formatted
    
    def _format_json_for_prompt_with_eight_counts(self, json_data):
        """Format JSON data for inclusion in the prompt with eight count information."""
        # Extract the essential information
        segments = json_data.get("segments", [])
        bpm = json_data.get("bpm", 120)  # Default to 120 if not specified
        
        # Calculate eight count duration in seconds (8 beats at given BPM)
        eight_count_duration_sec = (60.0 / bpm) * 8
        
        # Format for readability with both seconds and eight count information
        formatted = f"BPM: {bpm}\n\nSegments:\n"
        
        for i, segment in enumerate(segments):
            start_sec = segment["start"]
            end_sec = segment["end"]
            label = segment["label"]
            duration_sec = end_sec - start_sec
            
            # Convert to eight counts
            start_eight_count = start_sec / eight_count_duration_sec
            end_eight_count = end_sec / eight_count_duration_sec
            duration_eight_counts = duration_sec / eight_count_duration_sec
            
            # Format with both time measures
            formatted += (f"{i+1}. {label}: {start_sec:.2f}s - {end_sec:.2f}s "
                          f"({start_eight_count:.2f} - {end_eight_count:.2f} eight counts, "
                          f"duration: {duration_eight_counts:.2f} eight counts)\n")
        
        # Add explanation about eight counts
        formatted += "\nNote: One eight count equals 8 beats, standardized to 4 seconds at 120 BPM."
        
        return formatted
    
    def _select_segments(self, segments):
        """Select segments based on the selection strategy."""
        if len(segments) <= self.max_segments_per_song or self.segment_selection_strategy == "all":
            return segments
        
        if self.segment_selection_strategy == "sequential":
            # Take consecutive segments
            start_idx = random.randint(0, len(segments) - self.max_segments_per_song)
            return segments[start_idx:start_idx + self.max_segments_per_song]
        
        elif self.segment_selection_strategy == "random":
            # Randomly select segments
            return random.sample(segments, self.max_segments_per_song)
        
        # Default to sequential
        start_idx = random.randint(0, len(segments) - self.max_segments_per_song)
        return segments[start_idx:start_idx + self.max_segments_per_song]
    
    def __len__(self):
        return len(self.songs)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a training item with on-the-fly stem distortions and stem-specific tokens.
        
        Args:
            idx: Index of the song to retrieve
            
        Returns:
            Dictionary containing processed audio data, text prompts, and metadata
        """
        try:
            # Get the song info
            song = self.songs[idx]
            
            # Store song JSON data for beat-based chunking
            if "json_data" in song:
                self.current_song_json = song["json_data"]
            else:
                self.current_song_json = {}
            
            # Store BPM for beat-based chunking if available
            if "json_data" in song and "bpm" in song["json_data"]:
                self.current_song_bpm = song["json_data"]["bpm"]
            else:
                self.current_song_bpm = 120  # Default if not available
            
            # Select segments
            segments = self._select_segments(song["segments"])
            
            # Store all processed audio data
            processed_stems = {}
            original_stems = {}
            stem_actions = {}
            chunk_info = {}  # Store info about chunks for each stem/segment
            chunk_metadata = {}  # Store metadata for each chunk
            
            # Initialize loop counts tracking
            self.chunk_loop_counts = {}
            for stem_name in self.stem_types:
                self.chunk_loop_counts[stem_name] = []
                chunk_metadata[stem_name] = []
            
            # First pass: determine the maximum number of chunks per stem
            # This ensures we can properly align stems later
            max_chunks_per_stem = {}
            
            for stem_name in self.stem_types:
                stem_file = song["stem_files"][stem_name]
                max_chunks = 0
                
                # Count total chunks for this stem
                for segment_idx, segment in enumerate(segments):
                    # Extract segment from the stem - now returns a list of chunks with metadata
                    stem_chunks, loop_counts, chunk_metas = self._extract_segment_audio(stem_file, segment, stem_name=stem_name)
                    max_chunks += len(stem_chunks)
                
                max_chunks_per_stem[stem_name] = max_chunks
            
            # Find the maximum number of chunks across all stems
            # This ensures we can create tensors of the same size
            global_max_chunks = max(max_chunks_per_stem.values())
            
            # Create token identifiers for each stem and position
            stem_position_tokens = {}
            for stem_name in self.stem_types:
                stem_position_tokens[stem_name] = []
                
                # Create tokens like [STEM_POSITION]
                for i in range(global_max_chunks):
                    token = f"[{stem_name.upper()}_{i+1}]"
                    stem_position_tokens[stem_name].append(token)
            
            # Process each stem
            for stem_name in self.stem_types:
                stem_file = song["stem_files"][stem_name]
                
                # Initialize containers for this stem
                processed_stems[stem_name] = []
                original_stems[stem_name] = []
                stem_actions[stem_name] = []
                chunk_info[stem_name] = []
                
                # Store loop counts for each chunk
                if stem_name not in self.chunk_loop_counts:
                    self.chunk_loop_counts[stem_name] = []
                
                # Process each segment for this stem
                for segment_idx, segment in enumerate(segments):
                    # Extract segment from the stem - now returns a list of chunks, loop counts, and metadata
                    stem_chunks, loop_counts, chunk_metas = self._extract_segment_audio(stem_file, segment, stem_name=stem_name)
                    
                    # Track chunk info for this segment
                    chunk_info[stem_name].append(len(stem_chunks))
                    
                    # Store metadata for each chunk
                    for meta in chunk_metas:
                        # Add segment index to metadata
                        meta["segment_idx"] = segment_idx
                        chunk_metadata[stem_name].append(meta)
                    
                    # Process each chunk
                    for chunk_idx, chunk in enumerate(stem_chunks):
                        # Store loop count for this chunk
                        loop_count = loop_counts[chunk_idx] if chunk_idx < len(loop_counts) else 1
                        self.chunk_loop_counts[stem_name].append(loop_count)
                        
                        # Store original stem chunk
                        original_chunk_array = self._audio_to_numpy(chunk, self.sample_rate)
                        
                        # Ensure consistent length
                        if len(original_chunk_array) < self.fixed_audio_length:
                            # Pad with zeros if too short
                            padding = np.zeros(self.fixed_audio_length - len(original_chunk_array), dtype=np.float32)
                            original_chunk_array = np.concatenate([original_chunk_array, padding])
                        elif len(original_chunk_array) > self.fixed_audio_length:
                            # Trim if too long
                            original_chunk_array = original_chunk_array[:self.fixed_audio_length]
                        
                        original_stems[stem_name].append(original_chunk_array)
                        
                        # In training mode, create distorted version
                        if self.train_mode:
                            # Create distorter for this stem chunk
                            distorter = AudioDistorter(chunk, stem_name)
                            
                            # Apply distortions
                            distorted_chunk, actions = distorter.get_combined_distortions(self.num_distortions)
                            
                            # Store actions for this stem chunk
                            stem_actions[stem_name].append(actions)
                        else:
                            # In evaluation mode, just use original
                            distorted_chunk = chunk
                            stem_actions[stem_name].append([])
                        
                        # Convert to numpy array
                        distorted_chunk_array = self._audio_to_numpy(distorted_chunk, self.sample_rate)
                        
                        # Ensure consistent length
                        if len(distorted_chunk_array) < self.fixed_audio_length:
                            # Pad with zeros if too short
                            padding = np.zeros(self.fixed_audio_length - len(distorted_chunk_array), dtype=np.float32)
                            distorted_chunk_array = np.concatenate([distorted_chunk_array, padding])
                        elif len(distorted_chunk_array) > self.fixed_audio_length:
                            # Trim if too long
                            distorted_chunk_array = distorted_chunk_array[:self.fixed_audio_length]
                        
                        processed_stems[stem_name].append(distorted_chunk_array)
                
                # Pad stems to ensure all have the same number of chunks
                current_chunks = len(processed_stems[stem_name])
                if current_chunks < global_max_chunks:
                    # Create empty padding chunks
                    padding_chunk = np.zeros(self.fixed_audio_length, dtype=np.float32)
                    
                    # Add padding chunks for both processed and original stems
                    for _ in range(global_max_chunks - current_chunks):
                        processed_stems[stem_name].append(padding_chunk.copy())
                        original_stems[stem_name].append(padding_chunk.copy())
                        # Add empty actions for padding chunks
                        stem_actions[stem_name].append([])
                        
                        # Add padding metadata
                        padding_metadata = {
                            "stem_type": stem_name,
                            "segment_label": "padding",
                            "position": -1,
                            "segment_idx": -1,
                            "bpm": 120,
                            "speed_factor": 1.0,
                            "is_padding": True,
                            "start_sec": -1,
                            "end_sec": -1,
                            "start_eight_count": -1,
                            "end_eight_count": -1
                        }
                        chunk_metadata[stem_name].append(padding_metadata)
            
            # Generate text prompt
            if self.include_json:
                # Include JSON structure in the prompt with eight count information
                json_info = self._format_json_for_prompt_with_eight_counts(song["json_data"])
                
                # Randomly choose between traditional prompt and character-based prompt
                if random.random() < 0.7:  # 70% chance to use new mixer prompts
                    # Choose a mixer prompt
                    mixer_prompt = random.choice(self.mixer_prompts)
                    text_prompt = f"{mixer_prompt}\n\nSong Information:\n{json_info}\n\n"
                else:
                    # Use traditional prompt format
                    text_prompt = f"Fix and mix the following stems according to the song structure:\n\n{json_info}\n\n"
                
                # Add specific instructions for each stem
                for stem_name in self.stem_types:
                    all_stem_actions = []
                    for actions in stem_actions[stem_name]:
                        all_stem_actions.extend(actions)
                    
                    if all_stem_actions:
                        instructions = self._format_instructions(all_stem_actions)
                        text_prompt += f"{stem_name.upper()} STEM: {', '.join(instructions)}\n"
            else:
                # Use mixer prompts or a general prompt
                if random.random() < 0.7:  # 70% chance to use new mixer prompts
                    text_prompt = random.choice(self.mixer_prompts)
                else:
                    text_prompt = random.choice(self.prompt_templates[3:])  # Use general prompts
            
            # Generate tool tokens with chunk identifiers
            tool_tokens = []
            
            for stem_name in self.stem_types:
                chunk_idx = 0
                for segment_idx, num_chunks in enumerate(chunk_info[stem_name]):
                    segment_info = segments[segment_idx]
                    segment_label = segment_info["label"]
                    
                    # Process each chunk for this segment
                    for i in range(num_chunks):
                        if chunk_idx < len(stem_actions[stem_name]):
                            actions = stem_actions[stem_name][chunk_idx]
                            
                            # Get loop count for this chunk
                            loop_count = 1
                            if stem_name in self.chunk_loop_counts and chunk_idx < len(self.chunk_loop_counts[stem_name]):
                                loop_count = self.chunk_loop_counts[stem_name][chunk_idx]
                            
                            # Skip generating tokens for empty chunks (those with loop_count 0)
                            # Just add a single token indicating empty chunk
                            if loop_count == 0:
                                # Create identifier for this chunk
                                chunk_identifier = f"{stem_name}_{segment_label}_{segment_idx}_chunk{i}"
                                if chunk_idx < len(chunk_metadata[stem_name]):
                                    meta = chunk_metadata[stem_name][chunk_idx]
                                    # Use metadata for more detailed identifier if available
                                    if "position" in meta and "segment_label" in meta and "segment_idx" in meta:
                                        # Include eight count information if available
                                        if "start_eight_count" in meta and "end_eight_count" in meta:
                                            chunk_identifier = (f"{stem_name}_{meta['segment_label']}_{meta['segment_idx']}_"
                                                               f"pos{meta['position']}_"
                                                               f"eight_count{meta['start_eight_count']:.1f}-{meta['end_eight_count']:.1f}")
                                        else:
                                            chunk_identifier = f"{stem_name}_{meta['segment_label']}_{meta['segment_idx']}_pos{meta['position']}"
                                
                                # Generate and add token for empty/silent chunk
                                # Use a special empty/silent token in the ground truth
                                stem_token = f"<{stem_name}> <empty> <silent_chunk> <loop:0>"
                                tool_tokens.append(stem_token)
                                chunk_idx += 1
                                continue
                            
                            # Normal chunk processing
                            chunk_identifier = f"{stem_name}_{segment_label}_{segment_idx}_chunk{i}"
                            if chunk_idx < len(chunk_metadata[stem_name]):
                                meta = chunk_metadata[stem_name][chunk_idx]
                                # Use metadata for more detailed identifier if available
                                if "position" in meta and "segment_label" in meta and "segment_idx" in meta:
                                    # Include eight count information if available
                                    if "start_eight_count" in meta and "end_eight_count" in meta:
                                        chunk_identifier = (f"{stem_name}_{meta['segment_label']}_{meta['segment_idx']}_"
                                                           f"pos{meta['position']}_"
                                                           f"eight_count{meta['start_eight_count']:.1f}-{meta['end_eight_count']:.1f}")
                                    else:
                                        chunk_identifier = f"{stem_name}_{meta['segment_label']}_{meta['segment_idx']}_pos{meta['position']}"
                            
                            stem_token = self._generate_tool_tokens(
                                actions,
                                chunk_identifier,
                                loop_count
                            )
                            tool_tokens.append(stem_token)
                            chunk_idx += 1
            
            # Join all tool tokens
            combined_tool_tokens = " ".join(tool_tokens)
            
            # Tokenize text and tool tokens
            tokenized_prompt = self.tokenizer(
                text_prompt,
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt"
            )
            
            tokenized_tools = self.tokenizer(
                combined_tool_tokens,
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt"
            )
            
            # Convert all stems and chunks to tensors
            distorted_audio_tensors = []
            original_audio_tensors = []
            token_identifiers = []
            
            for stem_name in self.stem_types:
                # Stack all chunks for this stem
                if processed_stems[stem_name]:
                    stem_chunks = np.stack(processed_stems[stem_name])
                    original_chunks = np.stack(original_stems[stem_name])
                    
                    # Add stem-position tokens
                    stem_tokens = stem_position_tokens[stem_name][:len(stem_chunks)]
                    if len(stem_tokens) < len(stem_chunks):
                        # Pad with empty tokens if needed
                        stem_tokens = stem_tokens + ["[PAD]"] * (len(stem_chunks) - len(stem_tokens))
                    token_identifiers.append(stem_tokens)
                    
                    # Convert to tensor
                    distorted_audio_tensors.append(torch.tensor(stem_chunks, dtype=torch.float32))
                    original_audio_tensors.append(torch.tensor(original_chunks, dtype=torch.float32))
            
            # Stack stems - shape is [num_stems, num_chunks, audio_length]
            if distorted_audio_tensors:
                distorted_audio = torch.stack(distorted_audio_tensors)
                original_audio = torch.stack(original_audio_tensors)
                
                # Final check for NaN or Inf values
                distorted_audio = torch.nan_to_num(distorted_audio, nan=0.0, posinf=1.0, neginf=-1.0)
                original_audio = torch.nan_to_num(original_audio, nan=0.0, posinf=1.0, neginf=-1.0)
            else:
                # Fallback if no valid audio was processed
                dummy_audio = torch.zeros((len(self.stem_types), 1, self.fixed_audio_length), dtype=torch.float32)
                distorted_audio = dummy_audio
                original_audio = dummy_audio
                token_identifiers = [[f"[{stem.upper()}_1]"] for stem in self.stem_types]
            
            # Create segment and chunk metadata
            segment_metadata = []
            chunk_counter = 0
            
            for stem_idx, stem_name in enumerate(self.stem_types):
                for segment_idx, num_chunks in enumerate(chunk_info[stem_name]):
                    segment_info = segments[segment_idx] if segment_idx < len(segments) else {"label": "unknown"}
                    
                    # Process each chunk for this segment
                    for i in range(num_chunks):
                        if chunk_counter < len(token_identifiers[stem_idx]):
                            token_id = token_identifiers[stem_idx][chunk_counter]
                        else:
                            token_id = f"[{stem_name.upper()}_{chunk_counter+1}]"
                            
                        # Get metadata from chunk_metadata if available
                        meta = {}
                        if stem_name in chunk_metadata and i < len(chunk_metadata[stem_name]):
                            meta = chunk_metadata[stem_name][i]
                            
                        chunk_meta = {
                            "stem_name": stem_name,
                            "segment_label": segment_info["label"],
                            "segment_idx": segment_idx,
                            "chunk_idx": i,
                            "token_id": token_id,
                            **meta
                        }
                        segment_metadata.append(chunk_meta)
                        chunk_counter += 1
            
            # Prepare the item
            item = {
                "distorted_audio": distorted_audio,                   # [num_stems, num_chunks, audio_length]
                "original_audio": original_audio,                     # [num_stems, num_chunks, audio_length]
                "text_prompt": text_prompt,                           # String
                "input_ids": tokenized_prompt.input_ids.squeeze(0),   # [seq_len]
                "attention_mask": tokenized_prompt.attention_mask.squeeze(0), # [seq_len]
                "tool_input_ids": tokenized_tools.input_ids.squeeze(0),       # [seq_len]
                "tool_attention_mask": tokenized_tools.attention_mask.squeeze(0), # [seq_len]
                "segments": segment_metadata,                         # List of dict with chunk info
                "stem_actions": stem_actions,                         # Dict of list of list of dict
                "song_info": {
                    "artist": song["artist"],
                    "title": song["song_name"],
                    "full_path": song["full_wav"],
                    "json_path": song["json_file"]
                },
                "stem_types": self.stem_types,                        # List of str
                "tool_tokens": combined_tool_tokens,                  # String
                "token_identifiers": token_identifiers,               # List of list of str (stem position tokens)
                "encoded_audio": {
                    "distorted_shape": distorted_audio.shape,
                    "original_shape": original_audio.shape,
                    "distorted_mean": distorted_audio.mean().item(),
                    "original_mean": original_audio.mean().item(),
                    "distorted_std": distorted_audio.std().item(),
                    "original_std": original_audio.std().item(),
                }
            }
            
            return item
        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            traceback.print_exc()
            
            # If we hit any error, return a dummy item
            dummy_audio = torch.zeros((len(self.stem_types), 1, self.fixed_audio_length), dtype=torch.float32)
            dummy_token_ids = [[f"[{stem.upper()}_1]"] for stem in self.stem_types]
            
            return {
                "distorted_audio": dummy_audio,
                "original_audio": dummy_audio,
                "text_prompt": "Error processing song",
                "input_ids": torch.zeros((self.max_seq_len,), dtype=torch.long),
                "attention_mask": torch.zeros((self.max_seq_len,), dtype=torch.long),
                "tool_input_ids": torch.zeros((self.max_seq_len,), dtype=torch.long),
                "tool_attention_mask": torch.zeros((self.max_seq_len,), dtype=torch.long),
                "segments": [],
                "stem_actions": {stem: [[]] for stem in self.stem_types},
                "song_info": {
                    "artist": "error",
                    "title": "error",
                    "full_path": "",
                    "json_path": ""
                },
                "stem_types": self.stem_types,
                "tool_tokens": "",
                "token_identifiers": dummy_token_ids,
                "encoded_audio": {
                    "distorted_shape": dummy_audio.shape,
                    "original_shape": dummy_audio.shape,
                    "distorted_mean": 0.0,
                    "original_mean": 0.0,
                    "distorted_std": 0.0,
                    "original_std": 0.0,
                }
            }