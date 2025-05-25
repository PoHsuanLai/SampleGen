"""
Training infrastructure for the Hip-Hop Producer AI system.
"""

from .trainer import HipHopProducerTrainer, SelfSupervisedPretrainer
from .dataset import HipHopDataset, AudioDistortionDataset
from .mixer_dataset import MixerDataset
from .trainer import UnifiedProducerTrainer, FaustScriptTrainer
from .dataset import UnifiedProducerDataset, FaustScriptDataset
from .prompts import get_mixer_prompts, get_style_specific_prompts, get_random_mixing_prompt

__all__ = [
    "HipHopProducerTrainer",
    "SelfSupervisedPretrainer", 
    "HipHopDataset",
    "AudioDistortionDataset",
    "MixerDataset",
    "UnifiedProducerTrainer",
    "FaustScriptTrainer",
    "UnifiedProducerDataset",
    "FaustScriptDataset",
    "get_mixer_prompts",
    "get_style_specific_prompts", 
    "get_random_mixing_prompt"
] 