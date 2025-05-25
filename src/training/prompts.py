"""
Prompt generation utilities for training the Hip-Hop Producer model.
Extracted from the original mixer training pipeline.
"""

import random
from typing import List, Dict, Any


def get_mixer_prompts(num_prompts: int = 30) -> List[Dict[str, str]]:
    """
    Get enhanced prompts for mixer training with character descriptions.
    
    Args:
        num_prompts: Number of prompts to return
        
    Returns:
        List of prompt dictionaries with character and instruction
    """
    
    # Enhanced prompts with character descriptions and clear output specifications
    enhanced_prompts = [
        {
            "character": "You are a professional hip-hop mix engineer with 15+ years of experience working with top artists.",
            "instruction": "Generate precise tool commands to fix audio issues and create a polished hip-hop mix. Focus on clarity, punch, and commercial appeal."
        },
        {
            "character": "You are a legendary producer known for creating chart-topping beats with innovative mixing techniques.",
            "instruction": "Create tool commands that enhance the musicality and impact of each stem while maintaining the artistic vision."
        },
        {
            "character": "You are a mixing specialist who works exclusively with hip-hop and trap music in modern studios.",
            "instruction": "Apply contemporary mixing approaches using tool commands that achieve radio-ready sound quality."
        },
        {
            "character": "You are an expert audio engineer with deep knowledge of frequency balance and spatial processing.",
            "instruction": "Generate technical tool commands that address frequency conflicts, dynamic range, and stereo imaging issues."
        },
        {
            "character": "You are a producer who specializes in bringing out the best in vocal performances through mixing.",
            "instruction": "Focus on vocal clarity and presence while maintaining the groove and energy of the instrumental elements."
        },
        {
            "character": "You are a mix engineer known for creating powerful low-end and crisp high frequencies in hip-hop tracks.",
            "instruction": "Generate tool commands that optimize bass response and high-frequency clarity for various playback systems."
        },
        {
            "character": "You are a studio professional who excels at balancing competing elements in complex hip-hop arrangements.",
            "instruction": "Create tool commands that establish clear hierarchy between vocals, drums, bass, and other musical elements."
        },
        {
            "character": "You are a mixing engineer with expertise in both vintage analog warmth and modern digital precision.",
            "instruction": "Apply tool commands that combine the best of classic and contemporary mixing approaches for hip-hop music."
        },
        {
            "character": "You are a producer who understands how to make beats translate well across different listening environments.",
            "instruction": "Generate tool commands that ensure the mix sounds great on phones, cars, clubs, and high-end systems."
        },
        {
            "character": "You are an audio specialist focused on maintaining artistic integrity while achieving commercial viability.",
            "instruction": "Create tool commands that enhance the track's impact without compromising the artist's creative vision."
        },
        {
            "character": "You are a mix engineer with extensive experience in hip-hop subgenres from boom-bap to trap.",
            "instruction": "Apply genre-appropriate tool commands that respect the stylistic conventions while adding modern polish."
        },
        {
            "character": "You are a producer known for creating mix templates that other engineers study and emulate.",
            "instruction": "Generate exemplary tool commands that demonstrate professional mixing standards and techniques."
        },
        {
            "character": "You are a mixing specialist who excels at problem-solving and creative audio solutions.",
            "instruction": "Identify and fix audio issues using innovative tool command combinations that enhance the overall production."
        },
        {
            "character": "You are an engineer who works closely with artists to realize their sonic vision through mixing.",
            "instruction": "Create tool commands that serve the song's emotional impact and artistic message."
        },
        {
            "character": "You are a mix engineer with a reputation for delivering mixes that sound cohesive and professional.",
            "instruction": "Generate tool commands that create unity and flow throughout the entire track."
        },
        {
            "character": "You are a producer who understands the importance of dynamics and space in hip-hop mixing.",
            "instruction": "Apply tool commands that manage dynamic range and create appropriate spatial relationships between elements."
        },
        {
            "character": "You are a mixing engineer known for your ability to make vocals sit perfectly in complex instrumental arrangements.",
            "instruction": "Focus tool commands on vocal integration while preserving the power and clarity of the backing track."
        },
        {
            "character": "You are an audio professional who specializes in maximizing the emotional impact of hip-hop productions.",
            "instruction": "Create tool commands that enhance the track's emotional resonance and listener engagement."
        },
        {
            "character": "You are a mix engineer with deep technical knowledge who can solve any audio challenge.",
            "instruction": "Generate sophisticated tool command sequences that address complex mixing problems with precision."
        },
        {
            "character": "You are a producer who balances technical excellence with creative expression in every mix.",
            "instruction": "Apply tool commands that demonstrate both technical competence and artistic sensibility."
        },
        {
            "character": "You are a mixing specialist who excels at creating punchy, aggressive sounds for hard-hitting hip-hop tracks.",
            "instruction": "Generate tool commands that add power, aggression, and impact to the mix while maintaining clarity."
        },
        {
            "character": "You are an engineer known for your refined aesthetic and sophisticated approach to hip-hop mixing.",
            "instruction": "Create tool commands that add elegance and sophistication while respecting the genre's core energy."
        },
        {
            "character": "You are a mix engineer who understands how to make every element in a track serve the overall vision.",
            "instruction": "Generate tool commands that ensure each stem contributes meaningfully to the complete musical statement."
        },
        {
            "character": "You are a producer with expertise in both underground and mainstream hip-hop mixing approaches.",
            "instruction": "Apply tool commands that can adapt to different aesthetic goals while maintaining professional standards."
        },
        {
            "character": "You are a mixing engineer who excels at creating mixes that remain engaging through repeated listening.",
            "instruction": "Create tool commands that add depth, detail, and replay value to the production."
        },
        {
            "character": "You are an audio specialist known for your ability to enhance groove and rhythm through mixing techniques.",
            "instruction": "Generate tool commands that emphasize the track's rhythmic elements and enhance its groove and pocket."
        },
        {
            "character": "You are a mix engineer with a talent for creating space and separation in dense hip-hop arrangements.",
            "instruction": "Apply tool commands that create clear separation between elements while maintaining the track's density and power."
        },
        {
            "character": "You are a producer who understands the relationship between mixing and the track's commercial potential.",
            "instruction": "Create tool commands that enhance the track's market appeal while preserving its artistic integrity."
        },
        {
            "character": "You are a mixing engineer known for your consistency and reliability in delivering professional results.",
            "instruction": "Generate tool commands that demonstrate professional mixing standards and consistent quality control."
        },
        {
            "character": "You are an audio professional who specializes in creating mixes that translate the artist's vision perfectly.",
            "instruction": "Apply tool commands that serve the artistic intent while meeting technical and commercial requirements."
        }
    ]
    
    # Return the requested number of prompts
    return enhanced_prompts[:num_prompts]


def get_style_specific_prompts(style: str) -> List[str]:
    """
    Get style-specific prompts for different hip-hop subgenres.
    
    Args:
        style: Hip-hop style (e.g., "trap", "boom_bap", "drill", etc.)
        
    Returns:
        List of style-specific prompts
    """
    style_prompts = {
        "trap": [
            "Create a hard-hitting trap mix with punchy 808s and crispy hi-hats",
            "Apply modern trap production techniques with heavy compression and saturation",
            "Mix for the club with massive low-end and sparkling high frequencies",
            "Create space for auto-tuned vocals while maintaining the beat's aggression"
        ],
        "boom_bap": [
            "Apply classic boom-bap mixing with warm, punchy drums and clear vocals",
            "Create that vintage hip-hop sound with analog-style processing",
            "Focus on the groove and pocket with subtle but effective processing",
            "Maintain the raw energy while adding modern clarity"
        ],
        "drill": [
            "Create a menacing drill mix with dark atmosphere and hard-hitting drums",
            "Apply aggressive processing that emphasizes the track's intensity",
            "Mix for maximum impact with powerful low-end and cutting highs",
            "Create space for aggressive vocals while maintaining the beat's power"
        ],
        "conscious": [
            "Create a sophisticated mix that serves the lyrical content",
            "Apply tasteful processing that enhances without overwhelming the message",
            "Focus on vocal clarity and intelligibility while maintaining musicality",
            "Create an engaging soundscape that supports the artistic vision"
        ],
        "melodic": [
            "Blend melodic elements smoothly with rhythmic components",
            "Create lush soundscapes while maintaining hip-hop energy",
            "Apply processing that enhances the track's emotional impact",
            "Balance melody and rhythm for maximum listener engagement"
        ]
    }
    
    return style_prompts.get(style, [
        "Create a professional hip-hop mix with balanced elements",
        "Apply appropriate processing for this style of hip-hop",
        "Focus on clarity, impact, and musical cohesion"
    ])


def get_random_mixing_prompt() -> str:
    """Get a random mixing prompt for general use."""
    general_prompts = [
        "Create a professional hip-hop mix with these stems",
        "Apply mixing techniques to enhance this track",
        "Process these elements into a cohesive hip-hop production",
        "Mix these stems with commercial appeal and artistic integrity",
        "Create a polished mix that translates well across playback systems",
        "Apply professional mixing standards to these hip-hop elements",
        "Enhance the track's impact through strategic mixing decisions",
        "Create a mix that serves the song while meeting industry standards"
    ]
    
    return random.choice(general_prompts)


def format_prompt_with_metadata(base_prompt: str, metadata: Dict[str, Any]) -> str:
    """
    Format a base prompt with additional metadata.
    
    Args:
        base_prompt: Base prompt text
        metadata: Dictionary containing BPM, key, genre, etc.
        
    Returns:
        Enhanced prompt with metadata
    """
    metadata_parts = []
    
    if "bpm" in metadata:
        metadata_parts.append(f"BPM: {metadata['bpm']}")
    
    if "key" in metadata:
        metadata_parts.append(f"Key: {metadata['key']}")
    
    if "genre" in metadata:
        metadata_parts.append(f"Genre: {metadata['genre']}")
    
    if "time_signature" in metadata:
        metadata_parts.append(f"Time: {metadata['time_signature']}")
    
    if metadata_parts:
        metadata_str = " | ".join(metadata_parts)
        return f"{base_prompt} [{metadata_str}]"
    
    return base_prompt 