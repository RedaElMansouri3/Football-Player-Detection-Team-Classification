import numpy as np
import cv2
import tensorflow as tf
import os
from load_model import load

# --- CONFIGURATION ---
VIDEO_FILENAME = 'match_video.mp4'
OUTPUT_FOLDER = 'extracted_images'
CONFIDENCE_THRESHOLD = 0.5  # Only save clear detections
MAX_IMAGES = 300  # Stop after collecting this many images to save space

# Create folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Load the Google Model (We use this ONLY to find the people, not to classify them)
print("Loading model...")
detection_model, categories, category_index = load()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections, prediction_dict, tf.reshape(shapes, [-1])

cap = cv2.VideoCapture(VIDEO_FILENAME)
count = 0
frame_num = 0

print("Extracting images... (This might take a minute)")

while cap.isOpened() and count < MAX_IMAGES:
    ret, frame = cap.read()
    if not ret: break
    
    frame_num += 1
    
    # Process every 5th frame to get variety (avoiding duplicates)
    if frame_num % 5 != 0: continue

    h, w = frame.shape[:2]
    input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.float32)
    detections, _, _ = detect_fn(input_tensor)

    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = (detections['detection_classes'][0].numpy() + 1).astype(int)

    for i in range(len(scores)):
        if scores[i] > CONFIDENCE_THRESHOLD:
            if category_index.get(classes[i], {}).get('name') == 'person':
                # Get coordinates
                ymin = int(boxes[i][0] * h)
                xmin = int(boxes[i][1] * w)
                ymax = int(boxes[i][2] * h)
                xmax = int(boxes[i][3] * w)

                # Ensure box is valid
                if (ymax - ymin) > 20 and (xmax - xmin) > 10:
                    crop = frame[ymin:ymax, xmin:xmax]
                    
                    # Save the image
                    filename = os.path.join(OUTPUT_FOLDER, f"player_{count}.jpg")
                    cv2.imwrite(filename, crop)
                    count += 1
                    
                    if count >= MAX_IMAGES: break

    print(f"Collected {count} images...", end='\r')

cap.release()
print(f"\nDone! Saved {count} images to '{OUTPUT_FOLDER}'.")