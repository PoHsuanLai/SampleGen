import os
import mido
from typing import Dict, List, Optional, Tuple

def clean_midi(midi_path: str, output_path: Optional[str] = None) -> str:
    """
    Clean and normalize a MIDI file.
    
    Args:
        midi_path: Path to input MIDI file
        output_path: Optional path for output file (if None, overwrites input)
        
    Returns:
        Path to the cleaned MIDI file
    """
    if output_path is None:
        output_path = midi_path
        
    # Read the MIDI file
    midi_data = mido.MidiFile(midi_path)
    
    # Create a new MIDI file with cleaned tracks
    clean_midi = mido.MidiFile(ticks_per_beat=midi_data.ticks_per_beat)
    
    for i, track in enumerate(midi_data.tracks):
        # Create a new track
        new_track = mido.MidiTrack()
        
        # Process messages
        current_notes = {}  # to track note_on events without matching note_off
        
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                # Track this note
                current_notes[msg.note] = msg.time
            elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
                # Check if we have a matching note_on
                if msg.note in current_notes:
                    current_notes.pop(msg.note)
                    
            # Add the message to the new track
            new_track.append(msg)
            
        # Check for hanging notes and add note_off events
        for note, time in current_notes.items():
            # Add a note_off at the end
            end_time = sum(msg.time for msg in track if hasattr(msg, 'time'))
            new_track.append(mido.Message('note_off', note=note, velocity=0, time=end_time - time))
        
        # Add the track to the new MIDI file
        clean_midi.tracks.append(new_track)
    
    # Save the cleaned MIDI file
    clean_midi.save(output_path)
    
    return output_path

def quantize_midi(midi_path: str, output_path: Optional[str] = None, 
                 ticks_per_beat: int = 480, 
                 quantize_to: int = 16) -> str:
    """
    Quantize a MIDI file to a specific grid.
    
    Args:
        midi_path: Path to input MIDI file
        output_path: Optional path for output file (if None, overwrites input)
        ticks_per_beat: Ticks per quarter note
        quantize_to: Quantization level (e.g., 16 = 16th notes)
        
    Returns:
        Path to the quantized MIDI file
    """
    if output_path is None:
        output_path = midi_path
        
    # Read the MIDI file
    midi_data = mido.MidiFile(midi_path)
    
    # Create a new MIDI file
    quantized_midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    
    # Calculate the quantization grid in ticks
    grid_size = ticks_per_beat / quantize_to
    
    for track in midi_data.tracks:
        # Create a new track
        new_track = mido.MidiTrack()
        
        # Current time in ticks
        current_time = 0
        
        for msg in track:
            # Copy the message
            new_msg = msg.copy()
            
            if hasattr(msg, 'time'):
                # Update current time
                current_time += msg.time
                
                # Quantize the time
                quantized_time = round(current_time / grid_size) * grid_size
                
                # Set the delta time
                if len(new_track) == 0:
                    # First message, delta from 0
                    new_msg.time = int(quantized_time)
                else:
                    # Delta from previous message
                    prev_time = sum(m.time for m in new_track if hasattr(m, 'time'))
                    new_msg.time = int(quantized_time - prev_time)
            
            # Add message to the new track
            new_track.append(new_msg)
            
        # Add the track to the new MIDI file
        quantized_midi.tracks.append(new_track)
    
    # Save the quantized MIDI file
    quantized_midi.save(output_path)
    
    return output_path

def transpose_midi(midi_path: str, output_path: Optional[str] = None, semitones: int = 0) -> str:
    """
    Transpose a MIDI file by a number of semitones.
    
    Args:
        midi_path: Path to input MIDI file
        output_path: Optional path for output file (if None, overwrites input)
        semitones: Number of semitones to transpose (positive = up, negative = down)
        
    Returns:
        Path to the transposed MIDI file
    """
    if output_path is None:
        output_path = midi_path
        
    if semitones == 0:
        # No transposition needed
        return midi_path
        
    # Read the MIDI file
    midi_data = mido.MidiFile(midi_path)
    
    # Create a new MIDI file
    transposed_midi = mido.MidiFile(ticks_per_beat=midi_data.ticks_per_beat)
    
    for track in midi_data.tracks:
        # Create a new track
        new_track = mido.MidiTrack()
        
        for msg in track:
            # Copy the message
            new_msg = msg.copy()
            
            # Transpose note_on and note_off messages
            if msg.type in ('note_on', 'note_off') and hasattr(msg, 'note'):
                # Transpose the note
                new_note = msg.note + semitones
                
                # Ensure the note is within MIDI range (0-127)
                new_note = max(0, min(127, new_note))
                
                # Set the new note
                new_msg.note = new_note
            
            # Add message to the new track
            new_track.append(new_msg)
            
        # Add the track to the new MIDI file
        transposed_midi.tracks.append(new_track)
    
    # Save the transposed MIDI file
    transposed_midi.save(output_path)
    
    return output_path

def merge_midi_files(midi_paths: List[str], output_path: str) -> str:
    """
    Merge multiple MIDI files into a single file.
    
    Args:
        midi_paths: List of paths to input MIDI files
        output_path: Path for output merged file
        
    Returns:
        Path to the merged MIDI file
    """
    if not midi_paths:
        return None
        
    # Use the first file as a base
    first_midi = mido.MidiFile(midi_paths[0])
    merged_midi = mido.MidiFile(ticks_per_beat=first_midi.ticks_per_beat)
    
    # Add all tracks from all files
    for midi_path in midi_paths:
        midi_data = mido.MidiFile(midi_path)
        
        for track in midi_data.tracks:
            merged_midi.tracks.append(track)
    
    # Save the merged MIDI file
    merged_midi.save(output_path)
    
    return output_path 