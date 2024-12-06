import torch
import librosa
import logging
from speechbrain.pretrained import EncoderClassifier

# Load pre-trained model
model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_audio(audio_path):
    """Load and preprocess audio file."""
    try:
        audio, sr = librosa.load(audio_path, sr=16000)  # Load with 16kHz sampling rate
        return torch.tensor(audio).unsqueeze(0).to(device)
    except Exception as e:
        logging.error(f"Failed to preprocess audio: {audio_path}. Error: {e}")
        return None

def create_source_voiceprint_database(audio_paths):
    """Create a database of voiceprints from source audio."""
    source_voiceprints = {}
    for path in audio_paths:
        audio = preprocess_audio(path)
        if audio is not None:
            embedding = model.encode_batch(audio)
            source_voiceprints[path] = embedding
        else:
            logging.warning(f"Skipping file {path} due to preprocessing error.")
    return source_voiceprints

def calculate_average_target_voiceprint(audio_paths):
    """Calculate average voiceprint for similar speakers."""
    embeddings = []
    for path in audio_paths:
        audio = preprocess_audio(path)
        if audio is not None:
            embedding = model.encode_batch(audio)
            embeddings.append(embedding)
    if embeddings:
        return torch.mean(torch.stack(embeddings), dim=0)
    else:
        logging.error("No valid embeddings were created for similar speakers.")
        return None

def process_mixed_audio(audio_path, source_voiceprints, average_target_voiceprint):
    """Identify the source speaker from mixed audio."""
    audio = preprocess_audio(audio_path)
    if audio is None:
        logging.warning(f"Skipping file {audio_path} due to preprocessing error.")
        return None

    # Extract voiceprint and compare
    mixed_embedding = model.encode_batch(audio)
    max_similarity = float("-inf")
    identified_speaker = None

    for speaker, voiceprint in source_voiceprints.items():
        similarity = torch.nn.functional.cosine_similarity(
            mixed_embedding, voiceprint, dim=1
        ).mean().item()
        if similarity > max_similarity:
            max_similarity = similarity
            identified_speaker = speaker

    return {
        "Mixed Audio": audio_path,
        "Identified Speaker": identified_speaker,
        "Similarity": max_similarity,
    }

