
Voice Conversion Project
This project aims to identify the source speaker of mixed or converted audio files using a database of source voiceprints. The system utilizes machine learning and audio processing techniques to analyze voice characteristics and map them to the most probable source speaker.

Project Structure
voice_conversion_project/
├── data/
│   ├── source_audio/           # Original source speaker audio files
│   ├── mixed_audio/            # Converted/mixed audio files for identification
│   ├── similar_speakers/       # Audio of speakers similar to the source
├── scripts/
│   ├── main.py                 # Main pipeline script
│   ├── model.py                # Core model logic for voice processing
├── requirements.txt            # Python dependencies
├── results.csv                 # Output results of processing
├── venv/                       # Python virtual environment (if used)


Features
Source Voiceprint Database:


Extracts and stores voiceprint embeddings for source audio files.
Target Voiceprint Computation:


Computes an average voiceprint for similar speaker audio.
Mixed Audio Processing:


Identifies the closest matching source speaker for each mixed audio file.
Logging and Results:


Tracks processing progress and saves results in results.csv.

Requirements
Python 3.8+
CUDA-compatible GPU for faster processing (optional but recommended)

Setup Instructions
Clone the Repository:

 git clone https://github.com/aastha-sharma/voice_recovery_project.git
cd voice_conversion_project

Set Up the Environment:

 python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

How to Run
Activate the Virtual Environment:

 source venv/bin/activate
Run the Main Script:

 python3 scripts/main.py
View Results:


Check the results.csv file for identified speakers and similarity scores.
Example output:
 Mixed Audio,Identified Speaker,Similarity
data/mixed_audio/sample_1.wav,data/source_audio/speaker_01.wav,0.8742
data/mixed_audio/sample_2.wav,data/source_audio/speaker_03.wav,0.7321


Key Challenges
Device Mismatch Errors:


Resolved inconsistencies between CPU and GPU tensors.
Low Similarity Scores:


Improved embedding computation and logic to enhance accuracy.
Short/Noisy Audio Files:


Skipped insufficient audio files during processing.

Planned Improvements
Enhanced Embedding Models:


Experiment with advanced embedding architectures for better accuracy.
Noisy Audio Handling:


Integrate denoising techniques to handle noisy inputs.
Data Augmentation:




