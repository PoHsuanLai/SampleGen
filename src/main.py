# Import necessary libraries
import argparse
import os
import sys
from samplegen import SampleGen

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SampleGen: Generate music with LLM-orchestrated sampling and synthesis')
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to the input audio file')
    
    parser.add_argument('--output', '-o', type=str, default='generated_song.wav',
                       help='Path to save the output audio file (default: generated_song.wav)')
    
    parser.add_argument('--prompt', '-p', type=str, required=True,
                       help='Natural language prompt describing the desired music')
    
    parser.add_argument('--duration', '-d', type=float, default=30.0,
                       help='Maximum duration of the output in seconds (default: 30.0)')
    
    parser.add_argument('--stem-model', '-s', type=str, default='spleeter:5stems',
                       help='Stem separation model to use (default: spleeter:5stems)')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Initialize SampleGen
        samplegen = SampleGen(stem_model=args.stem_model)
        
        # Process the input
        summary = samplegen.process(
            input_audio_path=args.input,
            user_prompt=args.prompt,
            output_audio_path=args.output,
            duration=args.duration
        )
        
        # Print summary
        print("\nSUCCESS!")
        print(summary)
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()