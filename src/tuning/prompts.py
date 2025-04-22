"""
Prompts for music generation and finetuning specialized for hip-hop production.
This file contains curated text prompts organized by generator type and hip-hop style.
"""

from typing import Dict, List
import random

# Common hip-hop descriptors used across different prompt types
HIP_HOP_DESCRIPTORS = [
    "hard hitting", "booming", "bouncy", "groovy", "head nodding",
    "dirty", "raw", "smooth", "clean", "crisp", "tight"
]

# Style-specific terms for different hip-hop substyles
STYLE_TERMS = {
    "modern": ["trap influenced", "808 heavy", "digital", "minimalist", "modern hip-hop"],
    "classic": ["boom bap", "sample based", "90s style", "golden era", "classic hip-hop"],
    "trap": ["trap", "808 drums", "hi-hat rolls", "dark", "atmospheric", "southern"],
    "lofi": ["lo-fi", "mellow", "chill", "dusty", "vinyl", "relaxed"]
}

# Base prompts by generator type
BASE_PROMPTS = {
    "melody": [
        "hip-hop melody with catchy hooks",
        "rap beat melodic sample",
        "{style} hip-hop melodic motif",
        "urban melodic phrases for rap music",
        "sampled soul melody for hip-hop production",
        "synthesizer lead for rap beat",
        "{style} melodic hooks for rap",
        "catchy hip-hop piano melody",
        "loop-ready melody for rap production",
        "synthesizer melody for modern rap beat"
    ],
    
    "harmony": [
        "hip-hop chord progression",
        "{style} rap music harmony",
        "minor chord progression for rap beat",
        "jazzy hip-hop chords",
        "atmospheric harmony for rap production",
        "sample-like chords for hip-hop beat",
        "{style} harmonic background for rap",
        "dark chord progression for trap beat",
        "soulful hip-hop harmony",
        "moody keys for rap instrumental"
    ],
    
    "bass": [
        "heavy 808 bass line",
        "{style} hip-hop bass pattern",
        "booming rap bass",
        "sub bass for trap beat",
        "punchy bass line for hip-hop",
        "rhythmic bass groove for rap",
        "{style} bass drops for hip-hop",
        "clean bass line for rap instrumental",
        "deep sub bass for modern rap",
        "distorted bass for aggressive hip-hop"
    ],
    
    "drums": [
        "hard hitting hip-hop drum pattern",
        "{style} rap beat drums",
        "snappy snare and kick pattern for rap",
        "trap hi-hat pattern with 808 kicks",
        "boom bap drum break",
        "crisp hip-hop drum programming",
        "{style} drum rhythm for rap production",
        "layered drum pattern for hip-hop beat",
        "punchy kick and snare combo for rap",
        "classic sampled break feel for hip-hop"
    ]
}

# Mixer prompts components for structured generation

# 5 character variants
MIXER_CHARACTERS = [
    "You are a legendary mixing engineer with decades of experience in hip-hop production.",
    "You are a renowned sound designer specialized in crafting unique sonic textures.",
    "You are an award-winning producer known for pristine audio quality and innovative mixes.",
    "You are a veteran music producer who has worked with the biggest names in the industry.",
    "You are a hip-hop production specialist with expertise in creating powerful, commercial mixes."
]

# Many instruction variants
MIXER_INSTRUCTIONS = [
    "Create a professional mix with punchy bass and clear vocals.",
    "Restore the natural warmth to these stems while enhancing clarity.",
    "Design a mix with well-defined low end and sparkling highs, focusing on clarity and punch.",
    "Develop a minimalist mix by removing unnecessary elements and highlighting essential parts.",
    "Craft an aggressive, energetic mix that prioritizes impact while maintaining clarity.",
    "Create a boom bap style mix with vintage character and enhanced groove.",
    "Design a trap-influenced mix with hard-hitting 808s while maintaining space for vocals.",
    "Produce a warm, nostalgic lo-fi sound with tasteful imperfections.",
    "Balance all elements to create a sophisticated mix with clarity and depth.",
    "Design a commercial-ready urban mix with exceptional clarity and punch.",
    "Create a radio-ready mix that follows commercial loudness standards.",
    "Mix these stems for vinyl release, emphasizing analog warmth and manageable bass.",
    "Optimize this mix for streaming platforms with appropriate loudness targets.",
    "Develop a cinematic mix with wide stereo image and dramatic transitions.",
    "Create a mix optimized for club systems with controlled low end and punchy transients.",
    "Craft a modern trap beat with precise 808 tuning and rhythmic clarity.",
    "Design a classic boom bap mix with vinyl character and head-nodding groove.",
    "Mix in the style of Atlanta trap production with heavy sub bass and spatial atmospherics.",
    "Create a lo-fi hip-hop mix with warm textures and subtle imperfections.",
    "Produce a drill-style mix with sliding 808s and dark atmosphere.",
    "Fix phase issues and frequency masking problems in these stems.",
    "Address dynamic range issues for a consistent, balanced mix.",
    "Clean up unwanted frequencies while preserving essential character.",
    "Optimize stereo imaging for width while maintaining mono compatibility.",
    "Apply temporal effects strategically to create depth without muddiness.",
    "Prepare this mix for final release with appropriate loudness and frequency balance.",
    "Optimize each stem independently before final assembly.",
    "Enhance the clarity, presence, and emotion of vocal elements.",
    "Improve the groove and impact of the rhythm section.",
    "Maximize low-end impact while ensuring translation across playback systems."
]

# Consistent format template
MIXER_FORMAT = """
Format your response as follows:
1. DIAGNOSIS: List specific issues identified in each stem
2. PROCESSING CHAIN: For each stem, provide precise operations in this order:
   - Gain adjustments (dB values)
   - EQ settings (specific frequencies and gain)
   - Dynamics processing (threshold, ratio, attack, release)
   - Spatial effects (reverb time, delay values)
3. OUTPUT SETTINGS: Provide final output parameters (LUFS target, peak limiter settings)

Use exact numerical values that can be programmatically parsed.
"""

def generate_mixer_prompt():
    """
    Generate a structured mixer prompt with character, instruction, and format.
    
    Returns:
        A complete mixer prompt
    """
    character = random.choice(MIXER_CHARACTERS)
    instruction = random.choice(MIXER_INSTRUCTIONS)
    
    # Combine all three components
    return f"{character}\n\n{instruction}\n\n{MIXER_FORMAT}"

def get_mixer_prompts(num_prompts=5):
    """
    Get a collection of mixer prompts.
    
    Args:
        num_prompts: Number of prompts to return
        
    Returns:
        List of mixer prompts
    """
    prompts = []
    for _ in range(num_prompts):
        prompts.append(generate_mixer_prompt())
    return prompts

def get_hip_hop_prompts(generator_type: str, style: str = "modern") -> List[str]:
    """
    Get hip-hop specific prompts for the specified generator type and style.
    
    Args:
        generator_type: Type of generator (melody, harmony, bass, or drums)
        style: Hip-hop style (modern, classic, trap, lofi)
        
    Returns:
        List of text prompts for training or generation
    """
    if generator_type not in BASE_PROMPTS:
        raise ValueError(f"Unknown generator type: {generator_type}. Must be one of {list(BASE_PROMPTS.keys())}")
    
    if style not in STYLE_TERMS:
        style = "modern"  # Default to modern if unknown style
        
    base_prompts = BASE_PROMPTS[generator_type]
    style_specific = STYLE_TERMS[style]
    
    # Format any style placeholders in the base prompts
    formatted_base_prompts = [prompt.format(style=style) for prompt in base_prompts]
    
    # Combine with descriptors and style terms for more variation
    expanded_prompts = []
    for prompt in formatted_base_prompts:
        expanded_prompts.append(prompt)
        for desc in HIP_HOP_DESCRIPTORS:
            expanded_prompts.append(f"{desc} {prompt}")
        for term in style_specific:
            expanded_prompts.append(f"{term} {prompt}")
                
    return expanded_prompts


def get_custom_prompts(filepath: str, generator_type: str = None) -> List[str]:
    """
    Load custom prompts from a file.
    
    Args:
        filepath: Path to a text file with one prompt per line
        generator_type: Optional filter for generator-specific prompts
        
    Returns:
        List of custom prompts
    """
    try:
        with open(filepath, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
            
        # Filter by generator type if specified (assuming format "TYPE: prompt")
        if generator_type:
            filtered = []
            for prompt in prompts:
                if prompt.lower().startswith(f"{generator_type.lower()}:"):
                    filtered.append(prompt[len(generator_type)+1:].strip())
                elif not prompt.lower().split(':', 1)[0] in ["melody", "harmony", "bass", "drums"]:
                    filtered.append(prompt)  # Include untagged prompts
            return filtered
        return prompts
    except Exception as e:
        print(f"Error loading custom prompts: {str(e)}")
        return [] 