
# **Voice Recovery Project**

This project aims to identify the source speaker of mixed or converted audio files using a database of source voiceprints. The system utilizes machine learning and audio processing techniques to analyze voice characteristics and map them to the most probable source speaker.


## **Project Structure**

voice_conversion_project/
├── data_limited/
│   ├── source_audio/           # Limited original source speaker audio files
│   ├── mixed_audio/            # Limited converted/mixed audio files for identification
│   ├── similar_speakers/       # Limited audio of speakers similar to the source
├── scripts/
│   ├── main.py                 # Main pipeline script
│   ├── model.py                # Core model logic for voice processing
├── requirements.txt            # Python dependencies
├── results.csv                 # Output results of processing


## **Features**

### **Source Voiceprint Database**
- Extracts and stores voiceprint embeddings for source audio files.

### **Target Voiceprint Computation**
- Computes an average voiceprint for similar speaker audio.

### **Mixed Audio Processing**
- Identifies the closest matching source speaker for each mixed audio file.

### **Logging and Results**
- Tracks processing progress and saves results in `results.csv`.

---

## **Data Note**

Due to GitHub's file size and repository limits, **only a subset of the data** has been uploaded to the repository:
- Each folder (`source_audio`, `mixed_audio`, and `similar_speakers`) contains **up to 500 audio files** per subdirectory.
- For full-scale testing, replace the `data_limited` folder with your complete dataset structured in the same way.

---

## **Requirements**

- **Python 3.12.3+**
- **CUDA-compatible GPU** for faster processing (optional but recommended)

---

## **Setup Instructions**

1. **Clone the Repository:**

   ```
   git clone https://github.com/aastha-sharma/voice_recovery_project.git
   cd voice_recovery_project
   ```

2. **Set Up the Environment:**

   ```
   python3 -m venv venv
   
   source venv/bin/activate
   
   pip install --upgrade pip
   
   pip install -r requirements.txt
   
   ```

---

## **How to Run**

1. **Run the Main Script:**

   ```
   python3 scripts/main.py
   ```

3. **View Results:**

   Check the `results.csv` file for identified speakers and similarity scores.

   **Example output:**
   Mixed Audio,Identified Speaker,Similarity
   data_limited/mixed_audio/sample_1.wav,data_limited/source_audio/speaker_01.wav,0.8742
   data_limited/mixed_audio/sample_2.wav,data_limited/source_audio/speaker_03.wav,0.7321
   ```

## **Planned Improvements**

1. **Enhanced Embedding Models**
   - Experiment with advanced embedding architectures for better accuracy.

2. **Noisy Audio Handling**
   - Integrate denoising techniques to handle noisy inputs.

3. **Data Augmentation**
   - Use augmentation techniques to improve model generalization.

