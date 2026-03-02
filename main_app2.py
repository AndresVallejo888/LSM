
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns


clases_teclado = [
    "NEUTRAL", "A", "E", "I", "O", "U",          
    "ESPACIO", "BORRAR", "OK",                  
    "J", "K", "Q", "X", "Z",                    
    "B", "C", "D", "F", "G",                    
    "H", "L", "M", "N", "P",                    
    "R", "S", "T", "V", "W",                    
    "Y", "NADA"                                 
]


# Inicializar diccionario solo para las letras
conteo_detecciones = {letra: 0 for letra in clases_teclado if letra not in ["NEUTRAL", "NADA", "ESPACIO", "BORRAR", "OK"]}


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)


interpreter = tf.lite.Interpreter(model_path="modelo_teclado.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    if not ret: break
   
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands_detector.process(rgb)
   
    if res.multi_hand_landmarks:
        for hand_lms in res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
           
            # Filtro para evitar falsos positivos o atascamientos (como la "K")
            x_coords = [lm.x for lm in hand_lms.landmark]
            ancho_mano = max(x_coords) - min(x_coords)
           
            if ancho_mano < 0.05:
                continue
           
            base_x = hand_lms.landmark[0].x
            base_y = hand_lms.landmark[0].y
           
            coords = []
            for i in range(21):
                coords.append(hand_lms.landmark[i].x - base_x)
                coords.append(hand_lms.landmark[i].y - base_y)
           
            input_data = np.array([coords], dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]['index'])
           
            idx = np.argmax(out[0])
            prob_sena = out[0][idx]
           
            if idx < len(clases_teclado):
                nombre_sena = clases_teclado[idx]
            else:
                nombre_sena = f"Desconocido ({idx})"
           
            if prob_sena > 0.80:
                if nombre_sena in conteo_detecciones:
                    conteo_detecciones[nombre_sena] += 1
                color_texto = (0, 255, 0) # Verde si es mayor a 80%
            else:
                color_texto = (0, 0, 255) # Rojo si es menor a 80%
           
            cv2.putText(frame, f"Sena: {nombre_sena} ({prob_sena*100:.1f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_texto, 2)


    cv2.imshow("Evaluador de Modelo - Presiona 'q' para graficar", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break


cap.release()
cv2.destroyAllWindows()


# Generar gráfica al terminar
plt.figure(figsize=(14, 6))
sns.barplot(x=list(conteo_detecciones.keys()), y=list(conteo_detecciones.values()), palette="magma")
plt.title('Conteo de Detecciones por Letra (Facilidad de Detección)', fontsize=15)
plt.xlabel('Letra / Seña', fontsize=12)
plt.ylabel('Cantidad de veces detectada correctamente (>80%)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('conteo_detecciones.png')
plt.show()


