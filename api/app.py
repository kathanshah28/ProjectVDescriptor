# app.py
import cv2
import os
import time
import threading
import queue
import google.generativeai as genai
from dotenv import load_dotenv
from collections import deque
from flask import Flask, render_template, Response, jsonify
import atexit

# --- 1. INITIALIZATION AND SETUP ---

# Load environment variables
load_dotenv()
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("🔴 CRITICAL: GEMINI_API_KEY not found in .env file.")
    exit()

# --- FLASK APP AND GLOBAL RESOURCES ---
app = Flask(__name__)

# Video processing constants
VIDEO_CHUNK_DURATION = 8
TEMP_VIDEO_FILENAME = "temp_video_chunk.mp4"

# Thread-safe queues and shared data
video_chunk_queue = queue.Queue(maxsize=2)  # Max 2 chunks pending
caption_history = deque(maxlen=3)
caption_lock = threading.Lock()
stop_event = threading.Event()

# Initialize with a placeholder message
caption_history.append("Initializing system...")

# --- HELPER FUNCTION (from original script) ---
def wrap_text(text, font, font_scale, thickness, max_width):
    lines = []
    if not text:
        return lines
    words = text.split(' ')
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        (width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if width > max_width:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    lines.append(current_line)
    return lines

# --- 2. VIDEO PROCESSING WORKER (Functionality Preserved) ---
def video_processing_worker():
    """
    Pulls video chunks from the queue, sends them to Gemini, and updates
    the shared caption history. This runs in a parallel thread.
    """
    print("🟢 Video processing worker thread started.")
    model = genai.GenerativeModel('models/gemini-1.5-flash-latest')

    while not stop_event.is_set():
        try:
            frames, fps, frame_size = video_chunk_queue.get(timeout=1)
            print(f"🤖 Processing a {len(frames)}-frame chunk...")

            out = cv2.VideoWriter(TEMP_VIDEO_FILENAME, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
            for frame in frames:
                out.write(frame)
            out.release()

            video_file = genai.upload_file(path=TEMP_VIDEO_FILENAME)
            while video_file.state.name == "PROCESSING":
                time.sleep(1)
                video_file = genai.get_file(video_file.name)

            if video_file.state.name == "FAILED":
                raise ValueError("Gemini API: Video processing failed.")

            print("🧠 Generating description...")
            prompt = (
                "You are an expert scene describer for visually impaired users. "
                "Describe this video clearly and concisely. Focus on the main action, "
                "key objects, and any important text. Be objective and helpful."
            )
            response = model.generate_content([prompt, video_file], request_options={"timeout": 120})
            genai.delete_file(video_file.name)

            new_caption = response.text.strip().replace('\n', ' ')

            with caption_lock:
                caption_history.append(new_caption)
            print(f"🔊 New Caption: {new_caption}")
            video_chunk_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            error_message = f"An error occurred: {e}"
            print(f"🔴 {error_message}")
            with caption_lock:
                caption_history.append("Error during processing. Retrying...")
            time.sleep(2)

    print("🔴 Video processing worker thread finished.")

# --- 3. MAIN VIDEO STREAMING AND CHUNKING THREAD ---
def stream_generator():
    """
    Captures video, streams it to the web, and queues up chunks for processing.
    This is the main loop for the camera.
    """
    print("🟢 Starting camera and streaming...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("🔴 CRITICAL: Cannot open camera.")
        return

    # Start the background processing thread
    processing_thread = threading.Thread(target=video_processing_worker, daemon=True)
    processing_thread.start()

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)
    frames_in_chunk = fps * VIDEO_CHUNK_DURATION
    frame_buffer = []

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        # Add frame to buffer for chunking
        frame_buffer.append(frame.copy())
        if len(frame_buffer) >= frames_in_chunk:
            if not video_chunk_queue.full():
                video_chunk_queue.put((frame_buffer[:], fps, frame_size))
            frame_buffer = []

        # --- Draw captions on the frame ---
        with caption_lock:
            history_to_display = list(caption_history)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        margin = 15
        line_height = int(font_scale * 40)
        
        all_lines = []
        for text in history_to_display:
            lines = wrap_text(text, font, font_scale, thickness, frame_width - (2 * margin))
            all_lines.extend(lines)

        if all_lines:
            # Create a semi-transparent black background for the text
            bg_height = (len(all_lines) * line_height) + (2 * margin)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame_height - bg_height), (frame_width, frame_height), (0, 0, 0), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Draw text lines from bottom to top
            y = frame_height - margin
            for line in reversed(all_lines):
                (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
                cv2.putText(frame, line, (margin, y - text_height//2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                y -= line_height

        # Encode frame as JPEG and yield for streaming
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if not flag:
            continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

    cap.release()

# --- 4. FLASK WEB ROUTES ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Provides the video stream."""
    return Response(stream_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_caption')
def get_caption():
    """API endpoint for the browser to fetch the latest caption for TTS."""
    with caption_lock:
        latest = caption_history[-1] if caption_history else ""
    return jsonify(caption=latest)

@app.route('/about')
def about():
    return 'about page'

# --- 5. CLEANUP ---
def cleanup():
    """Gracefully stop all threads and clean up resources."""
    print("🔴 Shutting down server...")
    stop_event.set()
    if os.path.exists(TEMP_VIDEO_FILENAME):
        os.remove(TEMP_VIDEO_FILENAME)
    print("🟢 Cleanup complete.")

atexit.register(cleanup)

# if __name__ == '__main__':
#     print("🚀 Starting Live Scene Descriptor server...")
#     app.run(host='0.0.0.0', port=5000, threaded=True)