# app.py

import os
import time
import threading
import queue
import google.generativeai as genai
from dotenv import load_dotenv
from collections import deque
from flask import Flask, render_template, Response, jsonify, request
import atexit
import uuid

# --- 1. INITIALIZATION AND SETUP ---
load_dotenv()
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY_NEW"])
except KeyError:
    print("🔴 CRITICAL: GEMINI_API_KEY not found in .env file.")
    exit()

# --- FLASK APP AND GLOBAL RESOURCES ---
app = Flask(__name__)
# Create a directory to store temporary video chunks
TEMP_DIR = "temp_chunks"
os.makedirs(TEMP_DIR, exist_ok=True)

# The queue will now hold file paths of chunks to be processed
video_chunk_queue = queue.Queue(maxsize=5)
caption_history = deque(maxlen=10)
caption_lock = threading.Lock()
stop_event = threading.Event()
active_files = set() # Keep track of files being processed or recently created

caption_history.append("Ready. Point your camera at a scene.")

# --- 2. VIDEO PROCESSING WORKER (Modified to handle file paths) ---
def video_processing_worker():
    """
    Pulls a video file path from the queue, sends the file to Gemini,
    and updates the shared caption history.
    """
    print("🟢 Video processing worker thread started.")
    model = genai.GenerativeModel('models/gemini-1.5-flash-latest')

    while not stop_event.is_set():
        try:
            filepath = video_chunk_queue.get(timeout=1)
            print(f"🤖 Processing chunk: {os.path.basename(filepath)}")

            video_file = genai.upload_file(path=filepath)
            while video_file.state.name == "PROCESSING":
                time.sleep(1)
                video_file = genai.get_file(video_file.name)
                print(f"DEBUG: File '{os.path.basename(filepath)}' state: {video_file.state.name}, Error: {video_file.error_message if hasattr(video_file, 'error_message') else 'N/A'}")

            if video_file.state.name == "FAILED":
                # IMPROVE THIS LINE:
                error_detail = video_file.error_message if hasattr(video_file, 'error_message') and video_file.error_message else 'Unknown reason from Gemini API'
                print(f"🔴 Gemini API: Video processing failed for {os.path.basename(filepath)}. Reason: {error_detail}")
                raise ValueError("Gemini API: Video processing failed.")

            print("🧠 Generating description...")
            prompt = (
                "You are an expert scene describer for visually impaired users. "
                "Describe this video clearly and concisely. Focus on the main action, "
                "key objects, and any important text. Be objective and helpful."
            )
            response = model.generate_content([prompt, video_file], request_options={"timeout": 120})
            
            # Clean up the uploaded file on Gemini's side
            genai.delete_file(video_file.name)

            new_caption = response.text.strip().replace('\n', ' ')

            with caption_lock:
                caption_history.append(new_caption)
            print(f"🔊 New Caption: {new_caption}")
            
            # Clean up the local file after processing
            try:
                os.remove(filepath)
                active_files.discard(filepath)
            except OSError as e:
                print(f"🔴 Error deleting file {filepath}: {e}")

            video_chunk_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            error_message = f"An error occurred: {e}"
            print(f"🔴 {error_message}")
            with caption_lock:
                caption_history.append("Error during processing. Retrying...")
            # If a file was being processed, remove it from active set to allow cleanup later
            if 'filepath' in locals() and filepath in active_files:
                 active_files.discard(filepath)
            time.sleep(2)

    print("🔴 Video processing worker thread finished.")

# --- 3. FLASK WEB ROUTES ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    # Start the background processing thread when the first user connects
    global processing_thread
    if 'processing_thread' not in globals() or not processing_thread.is_alive():
        processing_thread = threading.Thread(target=video_processing_worker, daemon=True)
        processing_thread.start()
    return render_template('index.html')

@app.route('/upload_chunk', methods=['POST'])
def upload_chunk():
    """Receives a video chunk from the browser and queues it for processing."""
    if 'video-blob' not in request.files:
        return jsonify(status="error", message="no video blob"), 400

    video_blob = request.files['video-blob']
    
    # Generate a unique filename for the temporary chunk
    filename = f"{uuid.uuid4()}.webm"
    filepath = os.path.join(TEMP_DIR, filename)
    
    video_blob.save(filepath)
    active_files.add(filepath)
    
    # Add the path to the queue for the worker to process
    if not video_chunk_queue.full():
        video_chunk_queue.put(filepath)
        return jsonify(status="success", message="chunk queued")
    else:
        return jsonify(status="error", message="processing queue is full"), 503


@app.route('/get_captions')
def get_captions():
    """API endpoint for the browser to fetch the caption history."""
    with caption_lock:
        captions = list(reversed(caption_history))
    return jsonify(captions=captions)

# --- 4. CLEANUP ---
def cleanup():
    """Gracefully stop all threads and clean up resources."""
    print("🔴 Shutting down server...")
    stop_event.set()
    # Clean up any remaining temp files
    for f in list(active_files):
        try:
            os.remove(f)
        except OSError:
            pass
    if os.path.exists(TEMP_DIR):
        import shutil
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    print("🟢 Cleanup complete.")

atexit.register(cleanup)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5173, threaded=True)