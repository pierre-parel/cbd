import imutils
import math
from ultralytics import YOLO
import numpy as np
import cv2 as cv
import time
import os
import serial

# Set up serial communication with Arduino
arduino = serial.Serial(port="COM4", baudrate=9600, timeout=1)
time.sleep(2)  # Allow Arduino to initialize

# Import models
model_detect_top = YOLO("weights/best-detect-topbot.pt")   # Detection model for top camera
model_detect_bot = YOLO("weights/best-detect-topbot.pt")   # Detection model for bottom camera
model_cls_top = YOLO("weights/best-cls-top-v12.pt")           # Classification model for top camera
model_cls_bot = YOLO("weights/best-cls-top-v12.pt")        # Classification model for bottom camera

# Initialize cameras
vid0 = cv.VideoCapture(0)  # Bottom Camera
vid1 = cv.VideoCapture(1)  # Top Camera

# Directories for saving images
SAVE_DIR_MISCLASSIFIED = "Trials/Trial3"
os.makedirs(SAVE_DIR_MISCLASSIFIED, exist_ok=True)

# Accumulated results
results_summary = {
    "Total Beans Processed": 0,
    "Good Beans": 0,
    "Defective Beans": 0,
    "No Bean Detected": 0,
    "Bean Details": []
}

# Consecutive steps without bean detection
no_bean_counter = 0

def move_stepper():
    """Function to move the stepper motor."""
    print("🔄 Moving stepper motor...")
    arduino.write(b'NEXT_STEP\n')  # Send command to Arduino
    
    while True:
        data = arduino.readline().decode().strip()
        if data == "Step Completed":
            print("✅ Stepper move completed.")
            break
        time.sleep(0.2)  # Slight delay for better synchronization

def capture_and_process(camera, cam_number):
    """
    Captures and processes an image using the specified camera.
    Uses different classification models for top and bottom cameras.
    """
    ret, frame = camera.read()
    if not ret:
        print(f"❌ Failed to capture frame from Camera {cam_number}.")
        return None, None

    # Crop frame for better processing
    _, w, _ = frame.shape
    crop = 100
    frame_cropped = frame[:, crop:w-crop]

    # Select the correct model for detection and classification
    detect_model = model_detect_bot if cam_number == 0 else model_detect_top
    cls_model = model_cls_bot if cam_number == 0 else model_cls_top

    # Run detection model (Optimized inference for speed and stability)
    results = detect_model.predict(frame_cropped, conf=0.5, show=False, max_det=1)
    boxes = results[0].boxes.xyxy.tolist()

    # If no bean is detected, return None
    if len(boxes) != 1:
        print(f"⚠ No beans detected in Camera {cam_number}.")
        return frame, None  # Return the original frame for saving empty images

    # Extract detected region for classification
    x1, y1, x2, y2 = map(int, boxes[0])  # Convert to int
    im = frame_cropped[y1:y2, x1:x2]

    im = cv.resize(im, (640, 640))
    results_cls = cls_model.predict(im, show=False)
    result = results_cls[0].probs.top1  # Classify the bean

    width, height = boxes[0][2] - boxes[0][0], boxes[0][3] - boxes[0][1]
    size = [width, height]

    print(f"✅ Camera {cam_number} Classification: {result}")
    return frame, result, size  # Return the frame and classification

def save_misclassified(bean_index, frame, label, cam_position):
    """
    Saves an image of misclassified beans with labels.
    cam_position: 'Top' or 'Bot' (determines save directory)
    """
    if frame is not None:
        save_path = os.path.join(SAVE_DIR_MISCLASSIFIED, f"bean_{bean_index}_{cam_position}-{label}.jpg")
        cv.imwrite(save_path, frame)
        print(f"📸 Misclassified Screenshot saved: {save_path}")

def show_results():
    """Displays the final accumulated test results."""
    print("\n=== 📊 FINAL TEST RESULTS ===")
    print(f"Total Beans Processed: {results_summary['Total Beans Processed']}")
    print(f"✅ Good Beans: {results_summary['Good Beans']}")
    print(f"❌ Defective Beans: {results_summary['Defective Beans']}")
    print(f"⚠ No Bean Detected: {results_summary['No Bean Detected']}")
    print("\n🔎 Detailed Classifications:")
    
    for i, details in enumerate(results_summary["Bean Details"], start=1):
        print(f"  Bean {i}: Top - {details['Top']}, Bottom - {details['Bottom']}, Classification - {details['Final']}")

    print("\n🔚 Exiting program...")

def automated_testing():
    """Runs the full automated testing process continuously, with a stop prompt after 5 no-bean steps."""
    global no_bean_counter
    queue = []
    print("\n🚀 Starting Automated Testing... Press 'e' at any time to stop.\n")
    
    while True:
        if cv.waitKey(1) & 0xFF == ord('e'):
            print("\n🛑 Test manually stopped by the user.")
            break

        move_stepper()
        time.sleep(0.2)

        top_frame, top_result, top_size = capture_and_process(vid1, 1)
        bot_frame, bot_result, bot_size = capture_and_process(vid0, 0)

        if top_result is None or bot_result is None:
            no_bean_counter += 1
            results_summary["No Bean Detected"] += 1
            print("⚠ No bean detected in at least one camera. Skipping classification.\n")
            queue.append(-1)
            if no_bean_counter >= 20:
                choice = input("❓ Do you want to continue testing? (y/n): ").strip().lower()
                if choice == "n":
                    print("\n🛑 Stopping test early based on user input.")
                    break
                else:
                    no_bean_counter = 0
            if len(queue) == 4:
                val = queue.pop(0)
                if val == 0:
                    arduino.write(b'BAD_SERVO\n')
                elif val == 1:
                    arduino.write(b'GOOD_SERVO\n')
            continue
        else:
            no_bean_counter = 0

        results_summary["Total Beans Processed"] += 1

        if top_result == 5 and bot_result == 5:
            final_classification = "Good Beans"
            results_summary["Good Beans"] += 1
            queue.append(1)
            # arduino.write(b'GOOD_SERVO\n')
        else:
            final_classification = "Defective Beans"
            results_summary["Defective Beans"] += 1
            save_misclassified(results_summary["Total Beans Processed"], top_frame, top_result, "Top")
            save_misclassified(results_summary["Total Beans Processed"], bot_frame, bot_result, "Bot")
            # arduino.write(b'BAD_SERVO\n')
            queue.append(0)

        results_summary["Bean Details"].append({"Top": top_result, "Bottom": bot_result, "Final": final_classification})

        if len(queue) == 4:
            val = queue.pop(0)
            if val == 0:
                arduino.write(b'BAD_SERVO\n')
            elif val == 1:
                arduino.write(b'GOOD_SERVO\n')
        continue
    show_results()
    time.sleep(0.7)

def calculate_density(size):
    width, height = size
    depth = 0.4  # Assuming a fixed depth for simplicity
    if width == 0 or height == 0:
        return 0
    return math.round((math.pi/6) * width * height * depth / 1000, 2)  # Density in g/cm³

while True:
    key = input("\n🔹 Press 1 for Automated Testing, 2 for Test Capture (No Motor), 3 to Move Stepper, Q to Quit: ").strip().lower()
    
    if key == "1":
        automated_testing()

    elif key == "2":
        print("\n🔍 Running Test Capture (Without Motor Movement)...")
        _, top_result = capture_and_process(vid1, 1)
        _, bot_result = capture_and_process(vid0, 0)
        print(f"📸 Test Capture Results - Top: {top_result}, Bottom: {bot_result}\n")

        density_top = calculate_density(top_result) if top_result else 0
        density_bot = calculate_density(bot_result) if bot_result else 0
        print(f"📏 Density - Top: {density_top} g/cm³, Bottom: {density_bottom} g/cm³\n")

    elif key == "3":
        move_stepper()  # Move the stepper motor without processing
    
    elif key == "q":
        print("\n🛑 Test stopped by user.")
        show_results()
        break

vid0.release()
vid1.release()
cv.destroyAllWindows()