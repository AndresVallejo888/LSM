import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

archivo_csv = 'datasetLETRAS.csv'

# 1. Define aquí los nombres de todas tus señas
clases = [
    "NEUTRAL", 
    "A", 
    "E", 
    "I", 
    "O", 
    "U", 
    "ESPACIO", 
    "BORRAR", 
    "OK"
]


clase_actual = 8  

if not os.path.exists(archivo_csv):
    with open(archivo_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        encabezados = ['clase']
        for i in range(21):
            encabezados.extend([f'x{i}', f'y{i}'])
        writer.writerow(encabezados)

cap = cv2.VideoCapture(0)
grabando = False

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        for hand_landmarks in res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            if grabando:
                # Extraer coordenadas relativas a la muñeca (punto 0) para que no importe dónde está la mano en la pantalla
                base_x = hand_landmarks.landmark[0].x
                base_y = hand_landmarks.landmark[0].y
                
                fila = [clase_actual]
                for i in range(21):
                    x_relativo = hand_landmarks.landmark[i].x - base_x
                    y_relativo = hand_landmarks.landmark[i].y - base_y
                    fila.extend([x_relativo, y_relativo])
                
                with open(archivo_csv, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(fila)
                
                cv2.putText(frame, f"GRABANDO: {clases[clase_actual]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if not grabando:
        cv2.putText(frame, f"Pausado. Pulsa 'R' para grabar: {clases[clase_actual]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Recolector de Datos", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        grabando = not grabando
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()