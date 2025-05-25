"""
Unified Dataset for training the Unified Producer Model.
Focuses on stem generation planning and Faust script generation.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from torch.utils.data import Dataset
import soundfile as sf
import json
import random
import logging

# Import existing components
from ..data_processing.faust_distorter import FaustDistorter
from ..data_processing.stem_extraction import StemExtractor
from ..data_processing.audio_distorter import AudioDistorter

logger = logging.getLogger(__name__)


class UnifiedProducerDataset(Dataset):
    """
    Dataset for training the Unified Producer Model.
    Focuses on:
    1. Planning which stems to generate
    2. Creating appropriate generation prompts
    3. Generating Faust mixing scripts
    """
    
    def __init__(self,
                 data_dir: str,
                 sample_rate: int = 44100,
                 max_duration: float = 30.0,
                 use_distortion: bool = True,
                 segment_duration: float = 5.0,
                 max_segments_per_song: int = 3,
                 include_json: bool = True,
                 faust_script_training: bool = True):
        """
        Initialize dataset for unified producer training.
        
        Args:
            data_dir: Path to data directory containing artist folders
            sample_rate: Audio sample rate
            max_duration: Maximum duration per sample
            use_distortion: Whether to apply audio distortions
            segment_duration: Duration of audio segments in seconds
            max_segments_per_song: Maximum number of segments per song
            include_json: Whether to include JSON metadata in prompts
            faust_script_training: Whether to include Faust script generation training
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.use_distortion = use_distortion
        self.segment_duration = segment_duration
        self.max_segments_per_song = max_segments_per_song
        self.include_json = include_json
        self.faust_script_training = faust_script_training
        
        # Initialize stem extractor
        self.stem_extractor = StemExtractor(sample_rate=sample_rate)
        
        # Stem types to process
        self.stem_types = ["vocals", "drums", "bass", "other"]
        
        # Enhanced style prompts with JSON structure awareness
        self.enhanced_style_prompts = {
            'kendrick': "Conscious rap with complex rhythms, jazz samples, and intricate vocal layering",
            'travis': "Psychedelic trap with auto-tuned vocals, heavy reverb, and atmospheric production",
            'future': "Melodic trap with atmospheric pads, rolling hi-hats, and deep 808 patterns",
            'drake': "Smooth R&B-influenced hip-hop with emotional delivery and polished production",
            'kanye': "Innovative hip-hop with soul samples, creative arrangements, and bold sonic choices",
            'j_cole': "Lyrical hip-hop with live instrumentation, warm tones, and authentic production",
            'denzel_curry': "Aggressive rap with hard-hitting drums, distorted elements, and raw energy",
            'lil_uzi': "Melodic rap with ethereal synths, modern trap production, and catchy hooks",
            'pop_smoke': "Brooklyn drill with menacing bass, hard drums, and gritty urban atmosphere"
        }
        
        # Faust script templates for training
        self.faust_templates = {
            'bass_processing': [
                'bass_process = _ : fi.highpass(1, 40) : co.compressor_mono(4, -20, 0.003, 0.1);',
                'bass_process = _ : fi.lowpass(1, 120) : *(0.8) : co.compressor_mono(3, -18, 0.005, 0.15);',
                'bass_process = _ : fi.peak_eq(2, 80, 1.2) : co.compressor_mono(5, -22, 0.002, 0.08);'
            ],
            'drums_processing': [
                'drums_process = _ : fi.peak_eq(2, 100, 1) : co.compressor_mono(3, -15, 0.001, 0.05);',
                'drums_process = _ : fi.highpass(1, 60) : fi.peak_eq(1.5, 5000, 0.8) : co.compressor_mono(4, -12, 0.001, 0.03);',
                'drums_process = _ : fi.peak_eq(3, 200, 1.5) : fi.peak_eq(2, 8000, 1.2) : co.compressor_mono(6, -18, 0.0005, 0.02);'
            ],
            'melody_processing': [
                'melody_process = _ : fi.peak_eq(1.5, 2000, 0.7) : re.mono_freeverb(0.3, 0.5, 0.5, 0.5);',
                'melody_process = _ : fi.highpass(1, 200) : de.delay(ma.SR/4, 0.3) : re.mono_freeverb(0.2, 0.4, 0.6, 0.4);',
                'melody_process = _ : fi.peak_eq(1.2, 1500, 0.9) : co.compressor_mono(2, -25, 0.01, 0.2);'
            ],
            'harmony_processing': [
                'harmony_process = _ : fi.peak_eq(1.5, 2000, 0.7) : re.mono_freeverb(0.3, 0.5, 0.5, 0.5);',
                'harmony_process = _ : fi.lowpass(1, 8000) : re.mono_freeverb(0.4, 0.6, 0.7, 0.6) : *(0.6);',
                'harmony_process = _ : fi.peak_eq(0.8, 3000, 0.5) : de.delay(ma.SR/8, 0.2) : *(0.7);'
            ]
        }
        
        # Scan for songs with both audio and JSON metadata
        self.songs = self._scan_data_directory_with_json()
        
        print(f"Found {len(self.songs)} songs with complete metadata in {data_dir}")
    
    def _scan_data_directory_with_json(self) -> List[Dict]:
        """
        Scan the data directory to identify songs with JSON metadata for enhanced training.
        
        Returns:
            List of dictionaries containing song information with JSON metadata
        """
        songs = []
        
        # Find all available artists
        for artist_dir_name in os.listdir(self.data_dir):
            artist_path = os.path.join(self.data_dir, artist_dir_name)
            if not os.path.isdir(artist_path) or artist_dir_name.startswith('.'):
                continue
                
            # Find songs for each artist
            for song_dir_name in os.listdir(artist_path):
                song_dir = os.path.join(artist_path, song_dir_name)
                
                if not os.path.isdir(song_dir):
                    continue
                
                # Look for main audio file and JSON metadata
                wav_file = None
                json_file = None
                stems_dir = None
                
                for file in os.listdir(song_dir):
                    file_path = os.path.join(song_dir, file)
                    if file.endswith(".wav") and not os.path.isdir(file_path):
                        wav_file = file_path
                    elif file.endswith(".json"):
                        json_file = file_path
                    elif file == "stems" and os.path.isdir(file_path):
                        stems_dir = file_path
                
                # Skip if essential files are missing
                if not wav_file:
                    continue
                
                # Try to load JSON metadata if available
                json_data = None
                if json_file and os.path.exists(json_file):
                    try:
                        with open(json_file, 'r') as f:
                            json_data = json.load(f)
                    except Exception as e:
                        logger.warning(f"Could not load JSON for {song_dir_name}: {e}")
                
                # Check for pre-extracted stems
                stem_files = {}
                if stems_dir and os.path.exists(stems_dir):
                    for stem_file in os.listdir(stems_dir):
                        if stem_file.endswith(".wav"):
                            stem_name = os.path.splitext(stem_file)[0].lower()
                            if stem_name in self.stem_types:
                                stem_files[stem_name] = os.path.join(stems_dir, stem_file)
                
                songs.append({
                    "artist": artist_dir_name,
                    "song_name": song_dir_name,
                    "wav_file": wav_file,
                    "json_file": json_file,
                    "json_data": json_data,
                    "stem_files": stem_files,
                    "song_dir": song_dir
                })
        
        return songs
    
    def __len__(self) -> int:
        return len(self.songs)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, Dict]]:
        """Get a training sample for the unified producer model."""
        song_info = self.songs[idx]
        
        # Load audio data
        audio_data, sr = sf.read(song_info["wav_file"])
        if sr != self.sample_rate:
            # Simple resampling placeholder
            audio_data = audio_data
        
        # Extract segments if JSON data is available
        segments = []
        if song_info["json_data"] and "segments" in song_info["json_data"]:
            segments = song_info["json_data"]["segments"][:self.max_segments_per_song]
        
        # Extract stems (use pre-extracted if available, otherwise extract on-the-fly)
        if song_info["stem_files"]:
            stems = self._load_preextracted_stems(song_info["stem_files"], segments)
        else:
            stems = self._extract_stems_from_audio(audio_data, segments)
        
        # Create training targets
        training_data = self._create_training_targets(song_info, stems)
        
        # Apply distortions if enabled
        if self.use_distortion:
            distorted_stems, distortion_info = self._apply_distortions(stems)
            training_data['distorted_stems'] = distorted_stems
            training_data['distortion_info'] = distortion_info
        
        return training_data
    
    def _load_preextracted_stems(self, stem_files: Dict[str, str], segments: List[Dict]) -> Dict[str, np.ndarray]:
        """Load pre-extracted stems and optionally segment them."""
        stems = {}
        
        for stem_name, stem_path in stem_files.items():
            if stem_name in self.stem_types:
                stem_audio, sr = sf.read(stem_path)
                if len(stem_audio.shape) > 1:
                    stem_audio = np.mean(stem_audio, axis=1)  # Convert to mono
                
                # If segments are available, extract a random segment
                if segments:
                    segment = random.choice(segments)
                    start_sec = segment.get("start", 0)
                    end_sec = segment.get("end", len(stem_audio) / sr)
                    
                    start_sample = int(start_sec * sr)
                    end_sample = int(end_sec * sr)
                    
                    if start_sample < len(stem_audio) and end_sample > start_sample:
                        stem_audio = stem_audio[start_sample:end_sample]
                
                # Limit duration
                max_samples = int(self.max_duration * sr)
                if len(stem_audio) > max_samples:
                    start = random.randint(0, len(stem_audio) - max_samples)
                    stem_audio = stem_audio[start:start + max_samples]
                
                stems[stem_name] = stem_audio
        
        return stems
    
    def _extract_stems_from_audio(self, audio_data: np.ndarray, segments: List[Dict]) -> Dict[str, np.ndarray]:
        """Extract stems from audio using existing stem extractor."""
        # If segments are available, extract from a random segment
        if segments:
            segment = random.choice(segments)
            start_sec = segment.get("start", 0)
            end_sec = segment.get("end", len(audio_data) / self.sample_rate)
            
            start_sample = int(start_sec * self.sample_rate)
            end_sample = int(end_sec * self.sample_rate)
            
            if start_sample < len(audio_data) and end_sample > start_sample:
                audio_data = audio_data[start_sample:end_sample]
        
        # Limit duration
        max_samples = int(self.max_duration * self.sample_rate)
        if len(audio_data) > max_samples:
            start = random.randint(0, len(audio_data) - max_samples)
            audio_data = audio_data[start:start + max_samples]
        
        # Simple mock stems for now - in practice, use actual stem extraction
        stems = {
            'vocals': audio_data * 0.3,
            'drums': audio_data * 0.4,
            'bass': audio_data * 0.2,
            'other': audio_data * 0.1
        }
        
        return stems
    
    def _create_training_targets(self, song_info: Dict, stems: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Create training targets for the unified producer model."""
        # Create style prompt
        style_prompt = self._create_enhanced_style_prompt(song_info)
        
        # Create planning targets (which stems should be generated)
        planning_targets = self._create_planning_targets(stems, song_info)
        
        # Create generation prompts for each stem
        generation_prompts = self._create_generation_prompts(song_info, stems)
        
        # Create Faust script target if enabled
        faust_script = None
        if self.faust_script_training:
            faust_script = self._create_faust_script_target(stems, style_prompt)
        
        return {
            'stems': stems,
            'style_prompt': style_prompt,
            'planning_targets': planning_targets,
            'generation_prompts': generation_prompts,
            'faust_script': faust_script,
            'song_info': song_info
        }
    
    def _create_planning_targets(self, stems: Dict[str, np.ndarray], song_info: Dict) -> torch.Tensor:
        """Create planning targets indicating which stems should be generated."""
        # Create binary targets for each stem type
        stem_names = ['bass', 'drums', 'melody', 'harmony']
        targets = torch.zeros(4)
        
        # Map actual stem names to our standard names
        stem_mapping = {
            'vocals': 'melody',
            'drums': 'drums', 
            'bass': 'bass',
            'other': 'harmony'
        }
        
        # Set targets based on available stems and some randomness for augmentation
        for i, stem_name in enumerate(stem_names):
            # Check if we have this stem
            has_stem = any(stem_mapping.get(k) == stem_name for k in stems.keys())
            
            # Sometimes generate even if we have the stem (for enhancement)
            should_generate = not has_stem or random.random() < 0.3
            targets[i] = 1.0 if should_generate else 0.0
        
        return targets
    
    def _create_generation_prompts(self, song_info: Dict, stems: Dict[str, np.ndarray]) -> Dict[str, str]:
        """Create generation prompts for each stem type."""
        artist = song_info["artist"]
        base_style = self.enhanced_style_prompts.get(artist, "Hip-hop production")
        
        prompts = {
            'bass': f"{base_style} bass line with deep 808s and sub-bass frequencies",
            'drums': f"{base_style} drum pattern with hard-hitting kicks and crisp snares", 
            'melody': f"{base_style} melodic elements with catchy hooks and lead sounds",
            'harmony': f"{base_style} harmonic background with pads and atmospheric textures"
        }
        
        return prompts
    
    def _create_faust_script_target(self, stems: Dict[str, np.ndarray], style_prompt: str) -> str:
        """Create a target Faust script for training."""
        script_parts = ['import("stdfaust.lib");', '']
        
        # Add processing for each available stem
        available_stems = list(stems.keys())
        stem_mapping = {
            'vocals': 'melody',
            'drums': 'drums',
            'bass': 'bass', 
            'other': 'harmony'
        }
        
        for stem_name in available_stems:
            mapped_name = stem_mapping.get(stem_name, stem_name)
            if mapped_name in self.faust_templates:
                template_options = self.faust_templates[f'{mapped_name}_processing']
                chosen_template = random.choice(template_options)
                script_parts.append(chosen_template)
        
        # Add main process line
        if len(available_stems) > 1:
            process_inputs = ', '.join(['_'] * len(available_stems))
            process_functions = []
            for stem_name in available_stems:
                mapped_name = stem_mapping.get(stem_name, stem_name)
                process_functions.append(f'{mapped_name}_process')
            
            process_line = f'process = {process_inputs} : {", ".join(process_functions)} :> _;'
            script_parts.append('')
            script_parts.append(process_line)
        
        return '\n'.join(script_parts)
    
    def _apply_distortions(self, stems: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, List]]:
        """Apply distortions using Faust-based distorter."""
        distorted_stems = {}
        distortion_info = {}
        
        # Initialize Faust distorter
        faust_distorter = FaustDistorter(sample_rate=self.sample_rate)
        
        for stem_name, audio in stems.items():
            try:
                # Apply Faust-based distortion
                distorted_audio, correction_info = faust_distorter.apply_distortion(
                    audio, stem_type=stem_name
                )
                
                # Normalize distorted audio
                if len(distorted_audio) > 0 and np.max(np.abs(distorted_audio)) > 0:
                    distorted_audio = distorted_audio / np.max(np.abs(distorted_audio))
                
                distorted_stems[stem_name] = distorted_audio
                distortion_info[stem_name] = correction_info
                
            except Exception as e:
                # Fallback to original audio if distortion fails
                logger.warning(f"Faust distortion failed for {stem_name}: {e}")
                distorted_stems[stem_name] = audio
                distortion_info[stem_name] = {"tool": "no_change", "params": {}}
        
        return distorted_stems, distortion_info
    
    def _create_enhanced_style_prompt(self, song_info: Dict) -> str:
        """Create enhanced style prompt with JSON metadata integration."""
        artist = song_info["artist"]
        song_name = song_info["song_name"]
        
        # Base style from enhanced prompts
        base_style = self.enhanced_style_prompts.get(artist, "Hip-hop production")
        
        # Add JSON metadata if available
        metadata_info = ""
        if song_info["json_data"] and self.include_json:
            json_data = song_info["json_data"]
            
            # Add BPM information
            if "bpm" in json_data:
                metadata_info += f" at {json_data['bpm']} BPM"
            
            # Add key information
            if "key" in json_data:
                metadata_info += f" in {json_data['key']}"
            
            # Add genre information
            if "genre" in json_data:
                metadata_info += f", {json_data['genre']} style"
            
            # Add segment structure information
            if "segments" in json_data and len(json_data["segments"]) > 0:
                num_segments = len(json_data["segments"])
                metadata_info += f" with {num_segments} distinct sections"
        
        return f"{base_style}{metadata_info} in the style of {artist} - {song_name}"


class FaustScriptDataset(Dataset):
    """
    Specialized dataset for training Faust script generation.
    """
    
    def __init__(self,
                 audio_files: List[str],
                 sample_rate: int = 44100,
                 chunk_duration: float = 5.0):
        """
        Initialize Faust script dataset.
        
        Args:
            audio_files: List of audio file paths
            sample_rate: Audio sample rate
            chunk_duration: Duration of each training chunk
        """
        self.audio_files = audio_files
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        
        # Faust script templates for different scenarios
        self.script_templates = {
            'basic_mix': '''import("stdfaust.lib");
process = _, _, _, _ : 
    *(0.8), *(0.9), *(0.6), *(0.5) :> _;''',
            
            'compressed_mix': '''import("stdfaust.lib");
bass_comp = co.compressor_mono(4, -20, 0.003, 0.1);
drums_comp = co.compressor_mono(3, -15, 0.001, 0.05);
process = _, _, _, _ : 
    bass_comp, drums_comp, _, _ : 
    *(0.8), *(0.9), *(0.6), *(0.5) :> _;''',
            
            'eq_mix': '''import("stdfaust.lib");
bass_eq = fi.highpass(1, 40) : fi.peak_eq(2, 80, 1.2);
drums_eq = fi.peak_eq(2, 100, 1) : fi.peak_eq(1.5, 5000, 0.8);
melody_eq = fi.peak_eq(1.5, 2000, 0.7);
harmony_eq = fi.peak_eq(0.8, 3000, 0.5);
process = _, _, _, _ : 
    bass_eq, drums_eq, melody_eq, harmony_eq : 
    *(0.8), *(0.9), *(0.6), *(0.5) :> _;'''
        }
        
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """Get a Faust script training sample."""
        audio_file = self.audio_files[idx]
        
        # Create a simple prompt based on file path
        file_name = os.path.basename(audio_file)
        artist_hint = os.path.basename(os.path.dirname(audio_file))
        
        prompt = f"Create a Faust mixing script for {artist_hint} style hip-hop track"
        
        # Choose a random script template
        script_name = random.choice(list(self.script_templates.keys()))
        target_script = self.script_templates[script_name]
        
        return {
            'prompt': prompt,
            'target_script': target_script,
            'file_path': audio_file,
            'script_type': script_name
        } 