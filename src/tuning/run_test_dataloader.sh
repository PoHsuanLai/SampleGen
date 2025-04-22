#!/bin/bash
# Script to run the test_dataloader_standalone.py script with stem-level distortions and song structure
# Updated version:
# - Uses more aggressive distortions
# - Creates finer audio chunks, especially for bass and drums
# - Shows encoded audio information in output
# - Supports updated prompts from prompts.py

# Default parameters
DATA_DIR="./data"
NUM_SAMPLES=3
OUTPUT_DIR="./stem_test_output"

# Help message
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -d, --data-dir DIR      Path to the data directory (default: $DATA_DIR)"
    echo "  -n, --num-samples NUM   Number of samples to process (default: $NUM_SAMPLES)"
    echo "  -o, --output-dir DIR    Directory to save outputs (default: $OUTPUT_DIR)"
    echo "  -h, --help              Show this help message"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -n|--num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' does not exist."
    exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the test dataloader script
echo "Running test dataloader with parameters:"
echo "  Data directory: $DATA_DIR"
echo "  Number of samples: $NUM_SAMPLES"
echo "  Output directory: $OUTPUT_DIR"
echo "  Features: More aggressive distortions, finer audio chunks, encoded audio info"
echo

python src/tuning/test_dataloader_standalone.py \
    --data_dir "$DATA_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR"

echo
echo "Test complete. Results saved to $OUTPUT_DIR"
echo "You can view the generated audio files, visualizations, and encoded audio information in this directory." 