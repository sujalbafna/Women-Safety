import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import datetime

def faceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame, bboxs

# Load models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60+)']
genderList = ['Male', 'Female']

# Mediapipe setup for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Function to check for an open palm gesture (SOS situation)
def is_open_palm(hand_landmarks):
    # Compare finger tips and respective base knuckles
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y

    thumb_base = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y
    index_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
    middle_base = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
    ring_base = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
    pinky_base = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y

    # Ensure all fingers are extended (tips above base knuckles)
    if (thumb_tip < thumb_base and index_tip < index_base and
        middle_tip < middle_base and ring_tip < ring_base and
        pinky_tip < pinky_base):
        return True
    return False

# GUI setup
def update_frame():
    global male_count, female_count, female_surrounded, lone_female_detected, sos_detected
    ret, frame = video.read()
    if not ret:
        return
    
    frame, bboxs = faceBox(faceNet, frame)
    
    # Reset counts and flags
    male_count = 0
    female_count = 0
    female_surrounded = False
    lone_female_detected = False
    sos_detected = False
    
    for bbox in bboxs:
        face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]
        
        if gender == 'Male':
            male_count += 1
        else:
            female_count += 1
        
        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]
        
        label = "{},{}".format(gender, age)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Check for a female surrounded by males
    if female_count == 1 and male_count > 1:
        female_surrounded = True
    
    # Check for nighttime (7 PM to 6 AM) and lone female
    current_hour = datetime.datetime.now().hour
    if female_count == 1 and male_count == 0 and (current_hour >= 19 or current_hour <= 6):
        lone_female_detected = True
    
    # Hand detection for SOS
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Detect if an open palm (SOS gesture) is detected from a female
            if is_open_palm(hand_landmarks) and female_count == 1:
                sos_detected = True

    # Update GUI
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    
    male_label.config(text=f"Male: {male_count}")
    female_label.config(text=f"Female: {female_count}")
    
    if female_surrounded:
        female_surrounded_label.config(text="Female Surrounded by Males!", fg="red")
    else:
        female_surrounded_label.config(text="No Female Surrounded")
    
    if lone_female_detected:
        lone_female_label.config(text="Lone Female Detected at Night!", fg="red")
    else:
        lone_female_label.config(text="No Lone Female Detected")
    
    if sos_detected:
        sos_label.config(text="SOS Situation Detected!", fg="blue")
    else:
        sos_label.config(text="No SOS Detected")
    
    video_label.after(10, update_frame)

# Initialize counts and detection flags
male_count = 0
female_count = 0
female_surrounded = False
lone_female_detected = False
sos_detected = False

# Tkinter GUI initialization
root = tk.Tk()
root.title("Gender Detection with SOS and Surrounded Female Detection")

# Create a label for video feed
video_label = Label(root)
video_label.pack()

# Create labels for male and female counts
male_label = Label(root, text="Male: 0", font=('Helvetica', 12))
male_label.pack()
female_label = Label(root, text="Female: 0", font=('Helvetica', 12))
female_label.pack()

# Create a label for female surrounded detection
female_surrounded_label = Label(root, text="No Female Surrounded", font=('Helvetica', 14), fg="green")
female_surrounded_label.pack()

# Create a label for lone female detection
lone_female_label = Label(root, text="No Lone Female Detected", font=('Helvetica', 14), fg="green")
lone_female_label.pack()

# Create a label for SOS detection
sos_label = Label(root, text="No SOS Detected", font=('Helvetica', 14), fg="green")
sos_label.pack()

# Start video capture
video = cv2.VideoCapture(0)

# Update video frames and counts
update_frame()

# Run the GUI
root.mainloop()
