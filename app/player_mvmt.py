from flask import Flask, request, render_template, send_file
import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
OUTPUT_FOLDER = os.path.join(app.root_path, 'static')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO('../yolov8n.pt')  # Adjust path if necessary

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def process_video(video_path, output_video_path):
    print(f"Processing video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file!")
        return  # Exit function if the video can't be read

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    if fps == 0: 
        print("Error: FPS is 0, invalid video file!")
        return

    # Define output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Make sure format is correct
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print("Error: Failed to open VideoWriter!")
        return

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Finished processing {frame_count} frames.")
            break

        # Run YOLO detection
        results = model.track(frame, persist=True)
        
        # Ensure YOLO returned a valid result before processing
        if results:
            annotated_frame = results[0].plot()
            out.write(annotated_frame)  # Write processed frame

        frame_count += 1

    cap.release()
    out.release()
    
    print(f"Processing complete. Video saved at {output_video_path}")

# def process_video(video_path, output_video_path):
#     print(f"Processing video: {video_path}")
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise Exception("Error opening video file")

#     # Get video properties
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))

#     # Define output video writer
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

#     # Tracking data storage
#     player_positions = []  # List to store [frame, player_id, x, y]

#     frame_count = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Run YOLO detection
#         results = model.track(frame, persist=True)  # Enable tracking with persist=True

#         # Process detections
#         for result in results:
#             boxes = result.boxes
#             for box in boxes:
#                 if box.cls == 0:  # Class 0 is 'person' in COCO dataset
#                     x, y, w, h = box.xywh[0].tolist()  # Center x, y, width, height
#                     player_id = int(box.id) if box.id is not None else -1  # Tracking ID
#                     player_positions.append([frame_count, player_id, x, y])

#         # Draw results on frame
#         annotated_frame = results[0].plot()  # YOLO provides a plotting method
#         out.write(annotated_frame)
#         frame_count += 1

#     cap.release()
#     out.release()
#     print(f"Finished processing, saving to {output_video_path}")

#     # Generate movement plot
#     generate_movement_plot(player_positions, frame_width, frame_height)

def generate_movement_plot(positions, width, height):
    plt.figure(figsize=(10, 6))
    pitch_img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    plt.imshow(pitch_img)

    # Plot player paths
    for player_id in set(p[1] for p in positions if p[1] != -1):
        player_data = [p for p in positions if p[1] == player_id]
        x_coords = [p[2] for p in player_data]
        y_coords = [p[3] for p in player_data]
        plt.plot(x_coords, y_coords, label=f'Player {player_id}', marker='o', markersize=2)

    plt.title("Player Movement Tracks")
    plt.xlabel("X Position (pixels)")
    plt.ylabel("Y Position (pixels)")
    plt.legend()
    plt.savefig(os.path.join(app.config['OUTPUT_FOLDER'], 'movement_plot.png'))
    plt.close()

@app.route('/')
def index():
    return render_template('player_index.html')  # Use templates/player_index.html

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        original_video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(original_video_path)

        # Copy the original video to static for display
        static_original_path = os.path.join(app.config['OUTPUT_FOLDER'], 'original_' + filename)
        os.system(f'cp "{original_video_path}" "{static_original_path}"')

        # Process video for tracking
        output_video_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output.mp4')
        try:
            process_video(original_video_path, output_video_path)
            return render_template('result.html', 
                                # video_url='/static/output.mp4', 
                                video_url='/static/original_' + filename, 
                                plot_url='/static/movement_plot.png')
        except Exception as e:
            return f"Error processing video: {str(e)}", 500
    return "Invalid file type", 400

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return "No file part", 400
#     file = request.files['file']
#     if file.filename == '':
#         return "No selected file", 400
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         original_video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(original_video_path)
#         static_original_path = os.path.join(app.config['OUTPUT_FOLDER'], 'original_' + filename)
#         os.system(f'cp "{original_video_path}" "{static_original_path}"')

#         return render_template('result.html', 
#                                video_url='/static/original_' + filename, 
#                                plot_url='/static/movement_plot.png')

if __name__ == '__main__':
    app.run(debug=True)