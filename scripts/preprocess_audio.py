def preprocess_audio(audio_path):
    """
    Load and preprocess an audio file.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        torch.Tensor: Preprocessed audio tensor, or None if an error occurs.
    """
    try:
        # Load audio with a fixed sampling rate (16kHz)
        audio, sr = librosa.load(audio_path, sr=16000)

        # Normalize the audio to be in the range [-1, 1]
        audio = librosa.util.normalize(audio)

        # Convert the audio to a PyTorch tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

        # Move the tensor to the appropriate device (CPU or GPU)
        return audio_tensor.to(device)

    except Exception as e:
        logging.error(f"Failed to preprocess audio: {audio_path}. Error: {e}")
        return None
	
