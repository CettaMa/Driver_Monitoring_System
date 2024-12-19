import cv2
import dlib
import numpy as np

# Inisialisasi face detector dan facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('Landmarks/3_landmarks_model.dat')

# Fungsi untuk mengkonversi koordinat landmarks ke numpy array
def shape_to_np(shape):
    coords = np.zeros((27, 2), dtype=int)
    for i in range(0, 27):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# Memulai capture video
cap = cv2.VideoCapture(0)

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    
    # Compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = np.linalg.norm(eye[0] - eye[3])
    
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    
    return ear

def mouth_aspect_ratio(mouth):
    # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(mouth[1] - mouth[7])
    B = np.linalg.norm(mouth[2] - mouth[6])
    C = np.linalg.norm(mouth[3] - mouth[5])
    D = np.linalg.norm(mouth[0] - mouth[4])
    
    # Compute the mouth aspect ratio
    mar = (A + B + C) / (3.0*D)
    
    return mar
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # frame = cv2.imread('Landmarks/images.jpg')
    # Konversi frame ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Deteksi wajah
    faces = detector(gray)
    
    # Untuk setiap wajah yang terdeteksi
    for face in faces:
        # Prediksi landmarks
        landmarks = predictor(gray, face)
        landmarks = shape_to_np(landmarks)
        
        # Gambar landmarks
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
        # Gambar kotak di sekitar wajah
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Gambar bagian-bagian wajah tertentu dengan warna berbeda
        # Mulut dalam (31-39)
        cv2.polylines(frame, [landmarks[19:27]], True, (31, 219, 56), 1)
        # Mata kanan (13-19)
        cv2.polylines(frame, [landmarks[13:19]], True, (0, 255, 255), 1)
        # Mata kiri (7-13
        cv2.polylines(frame, [landmarks[7:13]], True, (0, 255, 255), 1)
        # Hidung (0-7)
        cv2.polylines(frame, [landmarks[0:7]], False, (219, 31, 31), 1)

    # Calculate Eye Aspect Ratio (EAR) for both eyes
        right_eye = landmarks[13:19]
        left_eye = landmarks[7:13]
        right_ear = eye_aspect_ratio(right_eye)
        left_ear = eye_aspect_ratio(left_eye)
        ear = (right_ear + left_ear) / 2.0
    # Calculate Mouth Aspect Ratio (MAR)
        mouth = landmarks[19:27]
        mar = eye_aspect_ratio(mouth)

        # Display EAR and MAR on the frame
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if ear < 0.23:
            cv2.putText(frame, "BLINK!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if mar > 0.65:
            cv2.putText(frame, "YAWN!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # Tampilkan frame
    cv2.imshow('Facial Landmarks Detection', frame)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()