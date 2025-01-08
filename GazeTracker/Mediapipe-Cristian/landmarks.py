import cv2
import mediapipe as mp

def get_landmarks_in_bbox(frame, face_landmarks, top_left_idx, bottom_right_idx):
    """
    Extract all landmarks within the bounding box defined by two landmarks.

    Args:
        frame: Input frame (for width and height scaling).
        face_landmarks: MediaPipe landmarks object for a single face.
        top_left_idx: Index of the top-left corner landmark.
        bottom_right_idx: Index of the bottom-right corner landmark.

    Returns:
        List of (x, y) coordinates of landmarks within the bounding box.
    """
    h, w, _ = frame.shape

    # Extract coordinates of the top-left and bottom-right corners
    x_min = int(face_landmarks.landmark[top_left_idx].x * w)
    y_min = int(face_landmarks.landmark[top_left_idx].y * h)
    x_max = int(face_landmarks.landmark[bottom_right_idx].x * w)
    y_max = int(face_landmarks.landmark[bottom_right_idx].y * h)

    # Ensure coordinates form a valid bounding box
    x_min, x_max = min(x_min, x_max), max(x_min, x_max)
    y_min, y_max = min(y_min, y_max), max(y_min, y_max)

    # Extract all landmarks within the bounding box
    landmarks_in_bbox = []
    for idx, lm in enumerate(face_landmarks.landmark):
        x = int(lm.x * w)
        y = int(lm.y * h)
        if x_min <= x <= x_max and y_min <= y <= y_max:
            landmarks_in_bbox.append((idx, x, y))

    return landmarks_in_bbox


def draw_eye_rectangles_and_landmarks(frame, landmarks, width, height):
    """
    Draw rectangles around the eyes and plot the landmarks within the bounding boxes.
    """
    # Define points for right and left eyes
    right_eye_points = [285, 261]
    left_eye_points = [46, 233]

    # Function to get bounding boxes and draw landmarks
    def process_eye(points, color):
        # Get bounding box
        xs = [landmarks.landmark[pt].x * width for pt in points]
        ys = [landmarks.landmark[pt].y * height for pt in points]
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))

        # Draw rectangle
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

        # Draw landmarks inside the bounding box
        for idx, lm in enumerate(landmarks.landmark):
            x = int(lm.x * width)
            y = int(lm.y * height)
            if x_min <= x <= x_max and y_min <= y <= y_max:
                cv2.circle(frame, (x, y), 1, color, -1)

    # Process right eye (blue)
    process_eye(right_eye_points, (255, 0, 0))

    # Process left eye (green)
    process_eye(left_eye_points, (0, 255, 0))


def main():
    # Initialize Mediapipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Unable to access the webcam.")
        return

    print("Press 'ESC' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame.")
            break

        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw rectangles and landmarks for the eyes
                draw_eye_rectangles_and_landmarks(frame, face_landmarks, width, height)

        cv2.imshow("Eye Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
