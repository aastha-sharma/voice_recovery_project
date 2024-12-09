# Voice Conversion Project

This project aims to **identify the source speaker** of mixed or converted audio files using a database of source voiceprints. The system utilizes **machine learning** and **audio processing techniques** to analyze voice characteristics and map them to the most probable source speaker.

## Project Structure

```plaintext
voice_conversion_project/
├── data/
│   ├── source_audio/           # Original source speaker audio files
│   ├── mixed_audio/            # Converted/mixed audio files for identification
│
├── data_limited/               # Limited demo files for quick testing
├── scripts/
│   ├── main.py                 # Main pipeline script
│   ├── model.py                # Core model logic for voice processing
    ├── preprocess_audio.py     # Preprocesses the audio files
├── requirements.txt            # Python dependencies
├── results.csv                 # Output results of processing
├── venv/                       # Python virtual environment (if used)
```

## Features

1. **Source Voiceprint Database**  
   - Extracts and stores voiceprint embeddings for source audio files.

2. **Target Voiceprint Computation**  
   - Computes an average voiceprint for similar speaker audio.

3. **Mixed Audio Processing**  
   - Identifies the closest matching source speaker for each mixed audio file.

4. **Logging and Results**  
   - Tracks processing progress and saves results in `results.csv`.

## Requirements

- **Python 3.12.3**
- **CUDA-compatible GPU** for faster processing (optional but recommended)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/aastha-sharma/voice_recovery_project.git
cd voice_recovery_project
```

### 2. Set Up the Environment
```bash
python3 -m venv venv
source venv/bin/activate      # On Windows use: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## About the `data_limited` Folder

The `data_limited` folder contains a subset of audio files for **demo purposes**. These files are intended for quick testing and demonstration on limited hardware.  
For the **full dataset** and to run the project with GPU acceleration, please contact me directly. I will provide access and further details on setting up the system.


## How to Run

### 1. Run the Main Script
```bash
python3 scripts/main.py
```

### 2. View Results
Check the `results.csv` file for identified speakers and similarity scores.

**Example Output:**

```csv
Mixed Audio,Identified Speaker,Similarity
data/mixed_audio/sample_1.wav,data/source_audio/speaker_01.wav,0.8742
data/mixed_audio/sample_2.wav,data/source_audio/speaker_03.wav,0.7321
```



## Key Challenges

1. **Device Mismatch Errors**  
   - Resolved inconsistencies between CPU and GPU tensors.

2. **Low Similarity Scores**  
   - Improved embedding computation and logic to enhance accuracy.

3. **Short/Noisy Audio Files**  
   - Skipped insufficient audio files during processing.



## Planned Improvements

1. **Enhanced Embedding Models**  
   - Experiment with advanced embedding architectures for better accuracy.

2. **Noisy Audio Handling**  
   - Integrate denoising techniques to handle noisy inputs.

3. **Data Augmentation**  
   - Add data augmentation techniques to improve generalization.



## Example Workflow

```
# Clone repository
git clone https://github.com/aastha-sharma/voice_recovery_project.git
cd voice_recovery_project

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the project
python3 scripts/main.py

# View results
cat results.csv
```
