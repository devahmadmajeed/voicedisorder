# 🎵 Audio Matching System

A Streamlit app that compares voice recordings with reference audio samples using Wav2Vec2 embeddings.

## Features

- Record or upload 5-second audio files
- Compare against reference audio samples (Male/Female, A/E/O vowels)
- Real-time similarity scoring
- Beautiful, user-friendly interface

## Deployment Guide

### Option 1: Streamlit Community Cloud (Recommended - Free & Easy)

1. **Push your code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository
   - Set Main file path: `app.py`
   - Click "Deploy"

3. **Your app will be live at:** `https://YOUR_APP_NAME.streamlit.app`

### Option 2: Deploy on Your Own Server

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run app.py
   ```

3. **For production, use:**
   ```bash
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```

## Requirements

- Python 3.8+
- All dependencies listed in `requirements.txt`
- Reference audio files (male_A.wav, male_E.wav, etc.) in the project root

## Local Setup

1. Create virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

## License

This project is open source and available for educational purposes.
