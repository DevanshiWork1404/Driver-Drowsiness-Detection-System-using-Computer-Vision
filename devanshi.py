import cv2
import time
import winsound

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not working")
    exit()

eye_closed_start = None
drowsy_threshold = 2
blink_ignore_time = 0.3
last_sound_time = 0
sound_gap = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    status = "ACTIVE"

    if len(faces) == 0:
        status = "NO FACE DETECTED"

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)

        current_time = time.time()

        if len(eyes) == 0:
            if eye_closed_start is None:
                eye_closed_start = current_time

            elapsed = current_time - eye_closed_start

            if elapsed > drowsy_threshold:
                status = "DROWSY ALERT!"
                if current_time - last_sound_time > sound_gap:
                    winsound.Beep(1000, 500)
                    last_sound_time = current_time

            elif elapsed > blink_ignore_time:
                status = "EYES CLOSED"
                if current_time - last_sound_time > sound_gap:
                    winsound.Beep(2000, 600)
                    last_sound_time = current_time

            else:
                status = "BLINKING"

        else:
            eye_closed_start = None
            status = "ACTIVE"

    cv2.putText(frame, status, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 3)

    cv2.imshow("Drowsiness Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()