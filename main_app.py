import os
import sys
import math
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import tensorflow as tf
import pyautogui
import screen_brightness_control as sbc

pyautogui.FAILSAFE = False 
screen_w, screen_h = pyautogui.size()

# --- DEFINICIÓN DE CLASES ---
clases_gestos = [
    "NEUTRAL", "MODO_CONFIG", "MODO_PANTALLA", "SUBIR_BRILLO", 
    "BAJAR_BRILLO", "SUBIR_VOLUMEN", "BAJAR_VOLUMEN", "CERRAR_VENTANA",
    "ABRIR_VENTANA", "MENU", "TECLADO"
]

clases_teclado = [
    "NEUTRAL", "A", "E", "I", "O", "U", 
    "ESPACIO", "BORRAR", "OK","J","K","Q","X","Z"
]

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except Exception as e:
    print(f"❌ Error al cargar MediaPipe: {e}")
    sys.exit()

# --- CARGAR AMBOS MODELOS ---
def cargar_modelo(ruta):
    interpreter = tf.lite.Interpreter(model_path=ruta)
    interpreter.allocate_tensors()
    return interpreter

try:
    int_gestos = cargar_modelo("modelo_gestos.tflite")
    int_teclado = cargar_modelo("modelo_teclado.tflite")
    print("✅ Modelos cargados con éxito")
except Exception as e:
    print(f"❌ Error con archivos del modelo: {e}")
    sys.exit()

hands_detector = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

estado = 'MENU_PRINCIPAL'
ultimo_comando = 0

print("🚀 Sistema de control por gestos iniciado. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands_detector.process(rgb)
    nombre_sena = "Ninguna"

    # Seleccionar qué modelo y qué lista de clases usar según el estado
    if estado == 'TECLADO':
        interpreter = int_teclado
        clases_actuales = clases_teclado
        cooldown_tiempo = 2.0  # 1.5 segundos de delay para escribir
    else:
        interpreter = int_gestos
        clases_actuales = clases_gestos
        cooldown_tiempo = 1.0  # 1 segundo de delay para navegar

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if res.multi_hand_landmarks:
        for hand_lms in res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # Predicción con coordenadas numéricas
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
            
            if np.max(out[0]) > 0.80:
                idx = np.argmax(out[0])
                nombre_sena = clases_actuales[idx]
            else:
                nombre_sena = "NEUTRAL"
            
            cv2.putText(frame, f"Sena: {nombre_sena} ({np.max(out[0])*100:.1f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            h, w, _ = frame.shape
            x_indice = int(hand_lms.landmark[8].x * w)
            y_indice = int(hand_lms.landmark[8].y * h)
            x_pulgar = int(hand_lms.landmark[4].x * w)
            y_pulgar = int(hand_lms.landmark[4].y * h)

    tiempo_actual = time.time()
    puede_ejecutar = (tiempo_actual - ultimo_comando) > cooldown_tiempo

    # --- MÁQUINA DE ESTADOS ---
    if estado == 'MENU_PRINCIPAL':
        cv2.putText(frame, "--- MENU PRINCIPAL ---", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        if puede_ejecutar:
            if nombre_sena == "MODO_CONFIG":
                estado = 'CONFIGURACION'
                ultimo_comando = tiempo_actual
            elif nombre_sena == "MODO_PANTALLA":
                estado = 'PANTALLA'
                ultimo_comando = tiempo_actual

    elif estado == 'CONFIGURACION':
        cv2.putText(frame, "--- MODO CONFIGURACION ---", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if puede_ejecutar:
            if nombre_sena == "SUBIR_BRILLO":
                try: sbc.set_brightness(min(100, sbc.get_brightness()[0] + 15))
                except: pass
                ultimo_comando = tiempo_actual
            elif nombre_sena == "BAJAR_BRILLO":
                try: sbc.set_brightness(max(0, sbc.get_brightness()[0] - 15))
                except: pass
                ultimo_comando = tiempo_actual
            elif nombre_sena == "SUBIR_VOLUMEN":
                pyautogui.press('volumeup', presses=5) 
                ultimo_comando = tiempo_actual
            elif nombre_sena == "BAJAR_VOLUMEN":
                pyautogui.press('volumedown', presses=5)
                ultimo_comando = tiempo_actual
            elif nombre_sena == "MENU":
                estado = 'MENU_PRINCIPAL'
                ultimo_comando = tiempo_actual

    elif estado == 'PANTALLA':
        cv2.putText(frame, "--- MODO PANTALLA ---", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        if res.multi_hand_landmarks:
            mouse_x = np.interp(x_indice, [0, w], [0, screen_w])
            mouse_y = np.interp(y_indice, [0, h], [0, screen_h])
            pyautogui.moveTo(mouse_x, mouse_y)
            
            if math.hypot(x_indice - x_pulgar, y_indice - y_pulgar) < 30:
                if (tiempo_actual - ultimo_comando) > 0.5: 
                    pyautogui.click()
                    cv2.putText(frame, "CLICK!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    ultimo_comando = tiempo_actual

        if puede_ejecutar:
            
            if nombre_sena == "TECLADO":
                estado = 'TECLADO'
                ultimo_comando = tiempo_actual 
            ## elif nombre_sena == "ABRIR_VENTANA":
            ##    pyautogui.hotkey('win', 'e') 
            ##    ultimo_comando = tiempo_actual
            elif nombre_sena == "MENU":
                estado = 'MENU_PRINCIPAL'
                ultimo_comando = tiempo_actual

    elif estado == 'TECLADO':
        cv2.putText(frame, "--- MODO TECLADO ---", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        cv2.putText(frame, "Escribiendo... Haz 'OK' para salir", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if puede_ejecutar and nombre_sena not in ["NEUTRAL", "Ninguna"]:
            if nombre_sena == "OK":
                estado = 'PANTALLA'
            elif nombre_sena == "ESPACIO":
                pyautogui.press('space')
            elif nombre_sena == "BORRAR":
                pyautogui.press('backspace')
            else:
                pyautogui.write(nombre_sena.lower())
            
            ultimo_comando = tiempo_actual

    if not puede_ejecutar:
        tiempo_restante = cooldown_tiempo - (tiempo_actual - ultimo_comando)
        cv2.putText(frame, f"Espera: {tiempo_restante:.1f}s", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Control PC por Gestos", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()