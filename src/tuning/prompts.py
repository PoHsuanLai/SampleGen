"""
Prompts for music generation and finetuning specialized for hip-hop production.
This file contains curated text prompts organized by generator type and hip-hop style.
"""

from typing import Dict, List

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