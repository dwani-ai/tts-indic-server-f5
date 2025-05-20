# TTS - Indic Server - IndicF5


## Features
- **Text-to-Speech (TTS)**: Generate audio from text in Indian languages using Parler TTS.

## Prerequisites
- **System Requirements - User **:
    - **Python**: 3.10
    - Ubuntu 22.04
    - git 
    - vscode
- **System Requirements - Server **:
  - Ubuntu with sufficient RAM (16GB+ recommended for models).
  - Optional: NVIDIA GPU with CUDA support for faster inference.
- **FFmpeg**: Required for audio processing (ASR).

- Server Setup

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/dwani-ai/tts-indic-server-f5
   cd tts-indic-server-f5
   ```

2. Install Libraries:
    - On Ubuntu: ```sudo apt-get install ffmpeg build-essential```

3. Set Up Virtual Environment:
    ```bash
    python -m venv venv
    source venv/bin/activate 
    ```

    ```bash
    pip install -r requirements.txt
    ```
    

5. Running the Server
    - Start the Server:
    ```bash
        python src/server/main.py --host 0.0.0.0 --port 7860 --config config_two
    ```

    - The server starts with models loaded on start
    - Access the interactive API docs at http://localhost:7860/docs.
        
    - Text-to-Speech: POST /v1/audio/speech
    ```  bash
    curl -X POST "http://localhost:7860/v1/audio/speech" -H "X-API-Key: your_secret_key" -H "Content-Type: application/json" -d '{"input": "ನಮಸ್ಕಾರ", "voice": "Female voice", "model": "ai4bharat/indic-parler-tts", "response_format": "mp3"}' --output speech.mp3
    ```

- Troubleshooting

    - Module Errors: Ensure all dependencies are installed. Re-run pip install if needed.
    FFmpeg Not Found: Install FFmpeg and ensure it’s in your PATH.
    Permission Denied: Run with sudo if accessing restricted ports (e.g., < 1024).


