import os
import sys

# Desactivar avisos de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# --- CONFIGURACIÓN DE MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# --- CARGAR MODELO TFLITE ---
try:
    interpreter = tf.lite.Interpreter(model_path="model_unquant2.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    class_names = [line.strip() for line in open("labels2.txt", "r").readlines()]
    print("✅ Todo cargado correctamente")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit()

# --- APP ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
estado = 'menu'
letras_disponibles = ["A", "B", "C"]
indice_letra = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    key = cv2.waitKey(1) & 0xFF

    if estado == 'menu':
        frame[:] = (40, 40, 40)
        cv2.putText(frame, "SISTEMA LSM - GUARDADO", (70, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "1. APRENDER | 2. PRACTICAR", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if key == ord('1'): estado = 'aprender'
        if key == ord('2'): estado = 'practicar'

    elif estado == 'aprender':
        letra = letras_disponibles[indice_letra]
        img_guia = cv2.imread(f"imagenes_lsm/{letra}.jpg")
        if img_guia is not None:
            img_guia = cv2.resize(img_guia, (200, 200))
            frame[50:250, 50:250] = img_guia
        cv2.putText(frame, f"LETRA: {letra}", (280, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        if key == ord('d'): indice_letra = (indice_letra + 1) % len(letras_disponibles)
        if key == ord('a'): indice_letra = (indice_letra - 1) % len(letras_disponibles)
        if key == ord('m'): estado = 'menu'

    elif estado == 'practicar':
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Prediccion
                img_input = cv2.resize(frame, (224, 224))
                img_input = np.expand_dims(img_input, axis=0).astype(np.float32)
                img_input = (img_input / 127.5) - 1
                
                interpreter.set_tensor(input_details[0]['index'], img_input)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                idx = np.argmax(output_data[0])
                prob = output_data[0][idx]

                if prob > 0.70:
                    nombre = "".join([i for i in class_names[idx] if not i.isdigit()]).strip()
                    cv2.putText(frame, f"{nombre} {int(prob*100)}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No detecto mano", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if key == ord('m'): estado = 'menu'

    cv2.imshow("LSM Pro", frame)
    if key == ord('q'): break

cap.release()
cv2.destroyAllWindows()