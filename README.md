# ProjectVDescriptor

A lightweight Flask web app that captures live camera footage, chunks it into short video segments, and sends each segment to Google's Gemini vision model for real-time scene description. The app is designed to assist visually impaired users by providing spoken or on-screen narration of the surrounding environment.

## Overview

ProjectVDescriptor combines:

- A browser-based live camera feed
- Video chunking from the user's device
- A Python Flask backend
- Gemini 1.5 Flash for scene understanding
- A caption panel with optional text-to-speech output

This project is intended as a demo or starting point for assistive, real-time visual description experiences.

## Features

- Starts the device camera from the browser
- Records short video chunks at regular intervals
- Uploads each chunk to the backend
- Sends the video to Gemini for automated scene description
- Maintains a recent caption history
- Displays captions in the UI
- Optionally reads the latest description aloud using browser speech synthesis
- Runs as a Flask app and is compatible with Vercel deployment

## Tech Stack

- Python
- Flask
- Google Generative AI
- HTML/CSS/JavaScript
- OpenCV support libraries
- Vercel deployment configuration

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── vercel.json
├── .env
├── .gitignore
├── temp_chunks/
└── templates/
    └── index.html
```

## How It Works

1. The browser requests camera access and starts recording a short video clip.
2. Each chunk is uploaded to the Flask server via `/upload_chunk`.
3. The server stores the chunk temporarily and enqueues it for processing.
4. A background worker processes the queued video file with Gemini.
5. The generated caption is added to a shared caption history.
6. The frontend polls `/get_captions` and displays the newest descriptions.
7. If audio is enabled, the browser speaks the newest caption aloud.

## Requirements

- Python 3.10+
- A modern browser with camera access enabled
- A Gemini API key

## Installation

1. Clone the repository:

```bash
git clone https://github.com/kathanshah28/ProjectVDescriptor.git
cd ProjectVDescriptor
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your Gemini API key:

```env
GEMINI_API_KEY_NEW=your_api_key_here
```

5. Run the app:

```bash
python app.py
```

Then open:

```text
http://localhost:5173
```

## Environment Variables

| Variable | Description |
| --- | --- |
| `GEMINI_API_KEY_NEW` | API key used to authenticate requests to the Gemini API |
