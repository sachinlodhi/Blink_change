import cv2
import mediapipe as mp
import random 
import numpy as np

LEFT_EYE = [33, 160, 158, 133, 153, 144]  # left eye static landmark
RIGHT_EYE = [362, 385, 387, 263, 373, 380]  # right eye static landmark

# Blink detection variables
BLINK_THRESHOLD = 0.2  # keeping it low for better detection(globalized)
BLINK_FRAMES = 3  # to very that the eye is closed for 3 frames
blink_counter = 0

def compute_ear(eye_points, landmarks): # EAR formual to detect the blink
    # Vertical distances
    h1 = np.linalg.norm(np.array(landmarks[eye_points[1]]) - np.array(landmarks[eye_points[5]]))
    h2 = np.linalg.norm(np.array(landmarks[eye_points[2]]) - np.array(landmarks[eye_points[4]]))
    # Horizontal distance
    w = np.linalg.norm(np.array(landmarks[eye_points[0]]) - np.array(landmarks[eye_points[3]]))
    return (h1 + h2) / (2.0 * w)

#object initialization
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Face mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
B,G,R = 0,0,0 # to store the random color for the lines when blink

cap = cv2.VideoCapture(0) # webcam(integrated)
# cv2.namedWindow("Face Mesh", cv2.WINDOW_NORMAL) 

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    frame = cv2.flip(frame, 1)
    # Convert the BGR image to RGB as MediaPipe uses RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get face mesh results
    results = face_mesh.process(rgb_frame)

    # Convert the image back to BGR for rendering
    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
   
    # Draw the face mesh annotations on the frame
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

            # Compute EAR for both eyes
            left_ear = compute_ear(LEFT_EYE, landmarks)
            right_ear = compute_ear(RIGHT_EYE, landmarks)
            avg_ear = (left_ear + right_ear) / 2  # Average EAR

            # Detect blink
            if avg_ear < BLINK_THRESHOLD:
                blink_counter += 1
                if blink_counter >= BLINK_FRAMES:
                    # cv2.putText(frame, "BLINK DETECTED!", (200, 50),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    B,G,R = random.randint(0,255), random.randint(0,255), random.randint(0,255)
            # Draw individual points with random colors
            for i, landmark in enumerate(face_landmarks.landmark):
                x, y = int(landmark.x * w), int(landmark.y * h)
                b, g, r = random.randint(0,255), random.randint(0,255), random.randint(0,255)
                cv2.circle(frame, (x+200, y), radius=1, color=(b, g, r), thickness=-1) # move right
                cv2.circle(frame, (x-200, y), radius=1, color=(b, g, r), thickness=-1) #move left
            
           # Iterate over all face landmarks and connect adjacent ones
            for i in range(len(face_landmarks.landmark) - 1):
                b, g, r = random.randint(0,255), random.randint(0,255), random.randint(0,255)
                x1, y1 = int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)
                x2, y2 = int(face_landmarks.landmark[i + 1].x * w), int(face_landmarks.landmark[i + 1].y * h)
                
                cv2.line(frame, (x1+200, y1), (x2+200, y2), (B,G,R), 1) # right
                cv2.line(frame, (x1-200, y1), (x2-200, y2), (abs(255-B),abs(255-G),abs(255-R)), 1) # left

   
        
        cv2.imshow("Face Mesh", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean up resources
cap.release()
cv2.destroyAllWindows()
