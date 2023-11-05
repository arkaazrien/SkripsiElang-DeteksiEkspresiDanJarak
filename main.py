import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from keras.models import load_model
from keras.utils import img_to_array
from keras.preprocessing import image
from tkinter import *
from tkinter import messagebox
import numpy as np
import time
import threading 

# Inisialisasi variabel
current_expression = None
start_time = 0  # Inisialisasi start_time dengan waktu saat ini
label = None  # Inisialisasi label
expressions_detected = []  # Inisialisasi list untuk menyimpan ekspresi yang terdeteksi
detection_interval = 60  # Jangka waktu deteksi (dalam detik, 1 menit)
reset_interval = 60  # Jangka waktu reset timer (dalam detik, 1 menit)
timer_duration = reset_interval  # Durasi awal timer
notification_shown = False  # Reset status notifikasi

# Inisialisasi dictionary untuk menghitung persentase setiap ekspresi
emotion_labels = ['Aman', 'Aman', 'Bahaya', 'Aman', 'Bahaya', 'Aman', 'Aman']
expression_percentages = {label: 0.0 for label in emotion_labels}

# Fungsi untuk mencetak ekspresi saat ini setiap detik
def print_current_expression():
    global current_expression
    while True:
        if current_expression is not None:
            print(f"Ekspresi saat ini: {current_expression}")
        time.sleep(1)  # Tunggu selama 1 detik

# Fungsi untuk menghitung kedalaman
def calculate_depth(img, detector):
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.3
        f = 700
        d = (W * f) / w
        cvzone.putTextRect(img, f'Depth: {int(d)}cm', 
                           (face[10][0] - 100, face[10][1] - 50), scale=2)
        if d <= 40:
            messagebox.showinfo("FACE RECOGNITION", 
                                "Bahaya, jangan terlalu dekat dengan layar")
    return img

# Fungsi untuk mengenali ekspresi
def recognize_emotion(img, classifier, emotion_labels):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_classifier.detectMultiScale(gray)
    label = None  # Inisialisasi label

    if len(faces_detected) == 1:
        for (x, y, w, h) in faces_detected:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48),
                                  interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                label_position = (x, y)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, label, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    elif len(faces_detected) > 1:
        cv2.putText(img, 'Multiple Faces Detected', (250, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(img, 'No Faces Detected', (250, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return img, label

# Fungsi untuk mengurangi timer
def update_timer():
    global timer_duration, notification_shown, current_expression, start_time

    current_time = time.time()
    if current_time - start_time >= 1:
        timer_duration -= 1
        start_time = current_time

    if timer_duration < 0:
        timer_duration = reset_interval
        if not notification_shown:
            if expression_percentages['Bahaya'] >= 70.0:
                messagebox.showerror("FACE RECOGNITION", 
                                     "Bahaya, segera beristirahat!")
            notification_shown = True

    notification_shown = False

# Fungsi untuk menampilkan persentase ekspresi Aman dan Bahaya
def display_expression_percentages(img):
    y_position = 20
    for emotion in ["Aman", "Bahaya"]:
        percentage = expression_percentages[emotion]
        text = f"{emotion}: {percentage:.2f}%"
        cv2.putText(img, text, (20, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_position += 30

# Fungsi untuk menampilkan timer
def display_timer(img, timer_duration):
    timer_text = f"Timer: {int(timer_duration)}s"
    cv2.putText(img, timer_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Fungsi utama
def main(classifier):
    global current_expression, start_time, label, expressions_detected, timer_duration,notification_shown

    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector(maxFaces=1)

    while True:
        _, img = cap.read()

        img = calculate_depth(img, detector)
        img, label = recognize_emotion(img, classifier, emotion_labels)

        if current_expression != label:
            print(f"Ekspresi {current_expression} telah berubah menjadi {label}.")
            current_expression = label

        if label is not None:
            expressions_detected.append(label)
            total_expressions = len(expressions_detected)
            for emotion in emotion_labels:
                count_emotion = expressions_detected.count(emotion)
                expression_percentages[emotion] = (count_emotion / total_expressions) * 100

        display_expression_percentages(img)
        update_timer()

        # Menampilkan timer pada gambar
        display_timer(img, timer_duration)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load the classifier model here if it's not already loaded
    classifier = load_model('Fix_150epoch_ bestXceptionPlusData2.h5')

    # Membuat thread untuk mencetak ekspresi saat ini
    print_thread = threading.Thread(target=print_current_expression)
    print_thread.daemon = True
    print_thread.start()

    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Call the main function with the loaded classifier
    main(classifier)
