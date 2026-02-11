def load_audio(file_path):
    # Function to load audio files
    import librosa
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

def save_audio(file_path, audio, sr):
    # Function to save audio files
    import soundfile as sf
    sf.write(file_path, audio, sr)

def trim_silence(audio, top_db=20):
    # Function to trim silence from audio
    import librosa
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed_audio

def extract_mfcc(audio, sr, n_mfcc=13):
    # Function to extract MFCC features from audio
    import librosa
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfccs

def audio_to_mel_spectrogram(audio, sr):
    # Function to convert audio to Mel spectrogram
    import librosa
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
    return mel_spectrogram