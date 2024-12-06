import logging
import csv
import os
from model import (
    create_source_voiceprint_database,
    calculate_average_target_voiceprint,
    process_mixed_audio,
)

def get_wav_files(directory):
    """Retrieve all .wav files from the directory."""
    wav_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    return wav_files

def save_results_to_csv(results, output_file="results.csv"):
    """Save the results to a CSV file."""
    with open(output_file, "w", newline="") as csvfile:
        fieldnames = ["Mixed Audio", "Identified Speaker", "Similarity"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

def main():
    # Paths to data_limited directories
    source_audio_dir = "data_limited/source_audio/"
    mixed_audio_dir = "data_limited/mixed_audio/"
    similar_speakers_dir = "data_limited/similar_speakers/"

    # Step 1: Create source voiceprint database
    logging.info("Step 1: Creating source voiceprint database...")
    source_audio_paths = get_wav_files(source_audio_dir)
    source_voiceprints = create_source_voiceprint_database(source_audio_paths)
    logging.info(f"Source voiceprint database created with {len(source_voiceprints)} entries.")

    # Step 2: Calculate average target voiceprint
    logging.info("Step 2: Calculating average target voiceprint...")
    similar_speakers_paths = get_wav_files(similar_speakers_dir)
    average_target_voiceprint = calculate_average_target_voiceprint(similar_speakers_paths)
    logging.info("Average target voiceprint computed.")

    # Step 3: Process mixed audio files
    logging.info("Step 3: Processing mixed audio files...")
    mixed_audio_paths = get_wav_files(mixed_audio_dir)
    results = []
    for index, path in enumerate(mixed_audio_paths):
        logging.info(f"Processing file {index + 1}/{len(mixed_audio_paths)}: {path}")
        try:
            result = process_mixed_audio(path, source_voiceprints, average_target_voiceprint)
            if result is not None:
                results.append(result)
                logging.info(f"File processed successfully: {path}")
                logging.info(f"Identified Speaker: {result['Identified Speaker']}, Similarity: {result['Similarity']:.4f}")
            else:
                logging.warning(f"No valid result for file: {path}")
        except Exception as e:
            logging.error(f"Failed to process file {path}. Error: {e}")

    # Save results to CSV
    save_results_to_csv(results)
    logging.info("Results saved to results.csv.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
