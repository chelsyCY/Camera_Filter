import cv2
import mediapipe as mp
import numpy as np

# Initialize filter
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=3 
)

# TURN OOOONNNN CAMERA 
cam = cv2.VideoCapture(0)
capture_count = 0
capture_feedback = False
feedback_frames = 0
frozen_frame = None

filters = {
    '1': cv2.imread("eye.png", cv2.IMREAD_UNCHANGED),
    '2': cv2.imread("crown.png", cv2.IMREAD_UNCHANGED),
    '3':cv2.imread("pink.png", cv2.IMREAD_UNCHANGED),
    '4':cv2.imread("pig.png",cv2.IMREAD_UNCHANGED)
}
current_filter_key = '1'


print("Press 'q' to quit")

ret, frame = cam.read()

h, w, _ = frame.shape 

filter_ui_positions = {
    '1': (w//2 - 180, h - 50), 
    '2': (w//2 - 60,  h - 50), 
    '3': (w//2 + 60,  h - 50), 
    '4': (w//2 + 180, h - 50) 
}

selected_color = (0, 255, 0)       # Green highlight
unselected_color = (255, 255, 255) # White border

def overlay_filter(frame, x_center, y_center, filter_img, scale=1.0):
    if filter_img is None:
        return
    
    h, w, _ = frame.shape
    filter_width = int(50 * scale)
    filter_height = int(filter_width * filter_img.shape[0]/filter_img.shape[1])

    x1 = int(x_center - filter_width / 2)
    y1 = int(y_center - filter_height / 2)
    x2 = x1 + filter_width
    y2 = y1 + filter_height

    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return

    resized_filter = cv2.resize(filter_img, (filter_width, filter_height))

    # Split BGR and alpha channels
    filter_bgr = resized_filter[:, :, :3]
    filter_mask = resized_filter[:, :, 3:] / 255.0

    # Blend onto frame
    frame[y1:y2, x1:x2] = (1 - filter_mask) * frame[y1:y2, x1:x2] + filter_mask * filter_bgr


def draw_filter_selector(frame):
    for key, pos in filter_ui_positions.items():
        x, y = pos
        icon = filters[key]

        # Draw circle bubble
        color = selected_color if key == current_filter_key else unselected_color
        cv2.circle(frame, (x, y), 40, color, 2)

        # place small icon
        if icon is not None:
            small = cv2.resize(icon, (60, 60))
            if small.shape[2] == 4:
                alpha = small[:, :, 3] / 255.0
                bgr = small[:, :, :3]
                roi = frame[y-30:y+30, x-30:x+30]
                blended = (1 - alpha[..., None]) * roi + alpha[..., None] * bgr
                frame[y-30:y+30, x-30:x+30] = blended
            else:
                # No transparency → normal paste
                frame[y-30:y+30, x-30:x+30] = small

    print(f"Switched to filter {current_filter_key}")

## add
def mouse_click(event, x, y, flags, param):
    global current_filter_key

    if event == cv2.EVENT_LBUTTONDOWN:
        for key, pos in filter_ui_positions.items():
            px, py = pos
            if (x - px) ** 2 + (y - py) ** 2 <= 40 ** 2:
                current_filter_key = key
                print(f"Switched to filter {key}")

cv2.namedWindow("Filter")
cv2.setMouseCallback("Filter", mouse_click)

def capturePicture(frame, capture_count):
    filename = f"capture_{capture_count}.png"
    cv2.imwrite(filename, frame)
    print(f"Saved: {filename}")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    filter_img = filters.get(current_filter_key)

    if results.multi_face_landmarks and filter_img is not None:

        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            filter_img = filters.get(current_filter_key)

            if filter_img is None:
                continue

            if current_filter_key =="1":
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]

                left_x, left_y = int(left_eye.x * w), int(left_eye.y * h)
                right_x, right_y = int(right_eye.x * w), int(right_eye.y * h)

                eye_distance = abs(right_x - left_x)
                scale = eye_distance / 60.0  

                overlay_filter(frame, left_x, left_y, filter_img, scale)
                overlay_filter(frame, right_x, right_y, filter_img, scale)
            elif current_filter_key =="2":
                    
                forehead = face_landmarks.landmark[10]
                forehead_x = int(forehead.x * w)
                forehead_y = int(forehead.y * h)

                left_face = face_landmarks.landmark[234]
                right_face = face_landmarks.landmark[454]

                face_width = abs(int(right_face.x * w) - int(left_face.x * w))
                scale = face_width / 120.0 *2.0  # Adjust scale for crown

                crown_y = forehead_y - int(20*scale)

                overlay_filter(frame, forehead_x, crown_y, filter_img, scale)
            elif current_filter_key == "3":
                forehead = face_landmarks.landmark[10]
                forehead_x = int(forehead.x * w)
                forehead_y = int(forehead.y * h)

                left_face = face_landmarks.landmark[234]
                right_face = face_landmarks.landmark[454]

                face_width = abs(int(right_face.x * w) - int(left_face.x * w))
                scale = face_width / 120.0 *1.0  # Adjust scale for crown

                crown_y = forehead_y - int(20*scale)
                overlay_filter(frame, forehead_x + 150, crown_y, filter_img, scale)
            elif current_filter_key == "4":
                nose = face_landmarks.landmark[1]
                nose_x = int(nose.x * w)
                nose_y = int(nose.y * h)

                left_face = face_landmarks.landmark[234]
                right_face = face_landmarks.landmark[454]
                face_width = abs(int(right_face.x * w) - int(left_face.x * w))
                scale = face_width / 120.0

                overlay_filter(frame, nose_x, nose_y, filter_img, scale)
            
    
    # Draw filter selection bubbles
    draw_filter_selector(frame)

    if capture_feedback and frozen_frame is not None:
        preview = cv2.resize(frozen_frame, (200, 120))
        x1, y1 = w-220, 20
        x2, y2=x1+200, y1+120
        cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (255, 255, 255), 2)
        frame[y1:y2, x1:x2] = preview
        feedback_frames -= 1
        if feedback_frames <= 0:
            capture_feedback = False 
    cv2.imshow("Filter", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        capture_count += 1
        capturePicture(frame, capture_count)
        frozen_frame = frame.copy()
        capture_feedback = True
        feedback_frames = 45

    elif chr(key) in filters.keys():
        current_filter_key = chr(key)
        print(f"Switched to filter {current_filter_key}")


cam.release()
cv2.destroyAllWindows()