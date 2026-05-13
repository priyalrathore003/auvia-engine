import os
import uuid
import asyncio
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import librosa
import soundfile as sf
import numpy as np

# Setup logging for the 'Chairman' level visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("auvia_engine")

app = FastAPI(title="Auvia Audio Engine - V1 Core")

# Use /tmp for cloud environments like Render
UPLOAD_DIR = "/tmp/processed_cache"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def apply_neural_timbre_correction(audio_path: str, output_path: str):
    """
    Core Logic: Spectral Shaping & Timbre Enhancement.
    Note: 'timbre' is the correct musical term.
    """
    y, sr = librosa.load(audio_path, sr=None)
    
    # Harmonic/Percussive separation
    y_harm = librosa.effects.harmonic(y, margin=3.0)
    
    # Spectral Shaping via STFT
    S = librosa.stft(y_harm)
    S_mag, S_phase = librosa.magphase(S)
    
    # Boost low-mid resonance for vocal warmth (150Hz - 500Hz)
    S_mag[0:20, :] *= 1.5 
    
    y_enhanced = librosa.istft(S_mag * S_phase)
    sf.write(output_path, y_enhanced, sr)

@app.post("/process-vocal/")
async def process_vocal(file: UploadFile = File(...)):
    # 1. Sanitize Filename (Security Fix)
    safe_filename = f"{uuid.uuid4()}.wav" 
    input_path = os.path.join(UPLOAD_DIR, f"in_{safe_filename}")
    output_path = os.path.join(UPLOAD_DIR, f"out_{safe_filename}")

    try:
        # 2. Async Write
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)

        # 3. Offload Sync Work to Thread (Performance Fix)
        # This prevents the server from 'freezing' during processing
        await asyncio.to_thread(apply_neural_timbre_correction, input_path, output_path)
        
        return FileResponse(path=output_path, filename=f"auvia_enhanced.wav")
    
    except Exception as e:
        logger.error(f"Processing Error: {e}")
        # 4. Obfuscate Internal Errors (Security Fix)
        raise HTTPException(status_code=500, detail="Internal Audio Processing Error.")
    
    finally:
        # 5. Auto-Cleanup (Ops Fix)
        if os.path.exists(input_path):
            os.remove(input_path)
        # Note: We keep the output file for the FileResponse, 
        # but in production, a background task would clear this after 1 hour.