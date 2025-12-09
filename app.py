import streamlit as st
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import torchaudio
import tempfile
import os
from pathlib import Path
import soundfile as sf
import numpy as np
import base64

# Initialize session state
if 'recorded_audio' not in st.session_state:
    st.session_state.recorded_audio = None
if 'audio_key' not in st.session_state:
    st.session_state.audio_key = 0

# Initialize model and processor
@st.cache_resource
def load_model():
    """
    Load Wav2Vec2 model and processor from local directory or HuggingFace.
    Uses Streamlit cache to avoid reloading on every rerun.
    
    Returns:
        tuple: (processor, model) - Wav2Vec2 processor and model instances
    """
    local_model_path = "./wav2vec2-base"
    
    try:
        if os.path.exists(local_model_path):
            st.info("Loading model from local directory...")
            processor = Wav2Vec2Processor.from_pretrained(local_model_path)
            model = Wav2Vec2Model.from_pretrained(local_model_path)
        else:
            st.info("Downloading model from HuggingFace (first time only)...")
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please run 'python download_model.py' first to download the model.")
        st.stop()

processor, model = load_model()

TARGET_SR = 16000
TARGET_LEN = int(5 * TARGET_SR)

def get_embed(audio_path):
    """
    Extract embedding from audio file using Wav2Vec2 model.
    
    Args:
        audio_path (str): Path to the audio file to process
        
    Returns:
        torch.Tensor: Audio embedding vector, or None if processing fails
    """
    try:
        try:
            audio_data, sr = sf.read(audio_path)
            if len(audio_data.shape) == 1:
                audio = torch.from_numpy(audio_data).float()
            else:
                audio = torch.from_numpy(audio_data.mean(axis=1)).float()
        except Exception:
            audio, sr = torchaudio.load(audio_path)
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0)
            else:
                audio = audio.squeeze(0)
        
        if sr != TARGET_SR:
            audio = torchaudio.functional.resample(audio, sr, TARGET_SR)
        
        if audio.shape[-1] > TARGET_LEN:
            audio = audio[:TARGET_LEN]
        else:
            padding = TARGET_LEN - audio.shape[-1]
            audio = torch.nn.functional.pad(audio, (0, padding))
        
        inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
        
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state
            hidden = hidden.mean(dim=1)
        
        return hidden
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

def calculate_similarity(emb1, emb2):
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        emb1 (torch.Tensor): First embedding vector
        emb2 (torch.Tensor): Second embedding vector
        
    Returns:
        float: Cosine similarity score between -1 and 1, or None if inputs are invalid
    """
    if emb1 is None or emb2 is None:
        return None
    sim = torch.nn.functional.cosine_similarity(emb1, emb2)
    return sim.item()

REFERENCE_AUDIOS = {
    "male": {
        "A": "male_A1.wav",
        "E": "male_E1.wav",
        "O": "male_O1.wav"
    },
    "female": {
        "A": "female_A.wav",
        "E": "female_E.wav",
        "O": "female_O.wav"
    }
}

# Streamlit UI
st.set_page_config(page_title="Audio Matching System", page_icon="🎵", layout="centered")

st.title("🎵 Audio Matching System")
st.markdown("---")

st.write("""
This app compares your voice recording with reference audio samples.
Record or upload a 5-second audio file and match it against our reference voices.
""")

# Step 1: Select Gender
st.subheader("Step 1: Select Gender")
gender = st.radio("Choose gender:", ["Male", "Female"], horizontal=True)

# Step 2: Select Vowel Sound
st.subheader("Step 2: Select Vowel Sound")
vowel = st.radio("Choose vowel sound:", ["A", "E", "O"], horizontal=True)

# Display selected reference
gender_lower = gender.lower()
st.info(f"📌 Reference: {gender} voice saying '{vowel}'")

# Play reference audio
reference_path = REFERENCE_AUDIOS[gender_lower][vowel]
if os.path.exists(reference_path):
    st.write("🔊 **Listen to the reference audio:**")
    with open(reference_path, 'rb') as audio_file:
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')
else:
    st.warning(f"⚠️ Reference audio file not found: {reference_path}")

# Step 3: Record or Upload Audio
st.subheader("Step 3: Record or Upload Your Audio")
st.write("⏱️ Duration: 5 seconds")

# Create tabs for recording vs uploading
tab1, tab2 = st.tabs(["🎤 Record Audio", "📁 Upload Audio"])

audio_data = None
audio_source = None

with tab1:
    st.write("🎙️ **Record your voice** - Recording will automatically stop after 5 seconds")
    
    # Use audio_recorder package with exact 5 second recording
    try:
        from audio_recorder_streamlit import audio_recorder
        
        # Add recording state management
        if 'is_recording' not in st.session_state:
            st.session_state.is_recording = False
        if 'recording_start_time' not in st.session_state:
            st.session_state.recording_start_time = None
        
        audio_bytes = audio_recorder(
            text="🎤 Click to Start 5-Second Recording",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            icon_name="microphone",
            icon_size="3x",
            sample_rate=16000,
            pause_threshold=5.0  # Auto-stop after 5 seconds
        )
        
        if audio_bytes and audio_bytes != st.session_state.recorded_audio:
            st.session_state.recorded_audio = audio_bytes
            st.success("✅ 5-second recording complete!")
            st.rerun()
            
    except ImportError:
        st.error("⚠️ Please install audio-recorder-streamlit: pip install audio-recorder-streamlit")
        st.info("Alternatively, use the 'Upload Audio' tab to upload a pre-recorded file.")
    
    # Display saved recording
    if st.session_state.recorded_audio is not None:
        audio_data = st.session_state.recorded_audio
        audio_source = "recorded"
        
        st.success("✅ Audio recorded successfully!")
        st.info("⏱️ Recording duration: 5 seconds (auto-trimmed/padded)")
        st.write("🔊 **Your recorded audio:**")
        st.audio(audio_data, format='audio/wav')
        
        # Button to re-record
        if st.button("🔄 Record Again", key="record_again"):
            st.session_state.recorded_audio = None
            st.rerun()

with tab2:
    uploaded_file = st.file_uploader(
        "Choose an audio file (WAV format recommended)",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        help="Upload your audio recording to compare with the reference"
    )
    
    if uploaded_file is not None:
        audio_data = uploaded_file.read()
        audio_source = "uploaded"
        st.success("✅ Audio file uploaded successfully!")
        st.write("🔊 **Your uploaded audio:**")
        st.audio(audio_data, format='audio/wav')

# Process and compare
if audio_data is not None:
    # Save audio data temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_data)
        tmp_path = tmp_file.name
    
    try:
        # Get reference audio path
        reference_path = REFERENCE_AUDIOS[gender_lower][vowel]
        
        # Check if reference file exists
        if not os.path.exists(reference_path):
            st.error(f"⚠️ Reference audio file not found: {reference_path}")
            st.info("Please ensure all reference audio files are in the correct location.")
        else:
            # Process button
            if st.button("🔍 Compare Audio", type="primary", key="compare_btn"):
                with st.spinner("Processing and comparing audio..."):
                    # Get embeddings
                    user_emb = get_embed(tmp_path)
                    ref_emb = get_embed(reference_path)
                    
                    # Calculate similarity
                    similarity = calculate_similarity(user_emb, ref_emb)
                    
                    if similarity is not None:
                        # Display results
                        st.markdown("---")
                        st.subheader("📊 Results")
                        
                        # Convert similarity to percentage
                        similarity_percentage = similarity * 100
                        
                        # Display similarity score
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Similarity Score", f"{similarity_percentage:.2f}%")
                        with col2:
                            st.metric("Match Quality", 
                                     "Excellent" if similarity_percentage > 80 else
                                     "Good" if similarity_percentage > 60 else
                                     "Fair" if similarity_percentage > 40 else "Poor")
                        
                        # Progress bar
                        st.progress(min(similarity_percentage / 100, 1.0))
                        
                        # Interpretation
                        st.markdown("### Interpretation")
                        if similarity_percentage > 80:
                            st.success("🎉 Excellent match! Your voice closely matches the reference.")
                        elif similarity_percentage > 60:
                            st.info("👍 Good match! Your voice has strong similarity to the reference.")
                        elif similarity_percentage > 40:
                            st.warning("🤔 Fair match. Consider adjusting your pronunciation.")
                        else:
                            st.error("❌ Low match. Try recording again with clearer pronunciation.")
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# Instructions
with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    1. **Select Gender**: Choose Male or Female
    2. **Select Vowel**: Choose the vowel sound (A, E, or O)
    3. **Listen**: Play the reference audio to hear how it should sound
    4. **Record or Upload**: 
       - **Record**: Click "Start Recording", speak for 5 seconds, then click "Save Recording"
       - **Upload**: Choose an audio file from your device
    5. **Compare**: Click the "Compare Audio" button to see your similarity score
    
    **Tips for best results:**
    - Record in a quiet environment
    - Speak clearly and pronounce the vowel sound distinctly
    - Keep the recording close to 5 seconds
    - Use WAV format for best quality
    """)

# Footer
st.markdown("---")
st.caption("Audio Matching System | Powered by Wav2Vec2")