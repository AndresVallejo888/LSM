import os
import sys
import math
import time
import cv2
import numpy as np
import tensorflow as tf
import pyautogui
import screen_brightness_control as sbc

# Configuración de entorno
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
pyautogui.FAILSAFE = False 
screen_w, screen_h = pyautogui.size()

# --- DEFINICIÓN DE CLASES (Basadas en los nuevos modelos) ---
clases_gestos = [
    "NEUTRAL", "MODO_CONFIG", "MODO_PANTALLA", "SUBIR_BRILLO", 
    "BAJAR_BRILLO", "SUBIR_VOLUMEN", "BAJAR_VOLUMEN", "CERRAR_VENTANA",
    "ABRIR_VENTANA", "MENU", "TECLADO", "MUTE", "SCREENSHOT"
]

clases_teclado = [
    "NEUTRAL", "A", "E", "I", "O", "U",          # 0-5
    "ESPACIO", "BORRAR", "OK",                   # 6-8
    "J", "K", "Q", "X", "Z",                     # 9-13
    "B", "C", "D", "F", "G",                     # 14-18
    "H", "L", "M", "N", "P",                     # 19-23
    "R", "S", "T", "V", "W",                     # 24-28
    "Y", "NADA"                                  # 29-30
]

# --- INICIALIZACIÓN DE MEDIAPIPE ---
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except Exception as e:
    print(f"❌ Error al cargar MediaPipe: {e}")
    sys.exit()

# --- FUNCIÓN PARA CARGAR MODELOS TFLITE ---
def cargar_modelo(ruta):
    try:
        interpreter = tf.lite.Interpreter(model_path=ruta)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        print(f"❌ Error con el archivo {ruta}: {e}")
        sys.exit()

# Carga de modelos avanzados del compañero
int_gestos = cargar_modelo("modelo_gestos.tflite")
int_teclado = cargar_modelo("modelo_teclado.tflite")
print("✅ Modelos de coordenadas cargados con éxito")

hands_detector = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
estado = 'MENU_PRINCIPAL'
ultimo_comando = 0

print("🚀 Sistema avanzado iniciado. Controla tu PC con señas.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands_detector.process(rgb)
    nombre_sena = "NEUTRAL"
    prob_sena = 0

    # Selección de modelo según el estado   
    if estado == 'TECLADO':
        interpreter = int_teclado
        clases_actuales = clases_teclado
        cooldown_tiempo = 2.0  # Delay para escritura
    else:
        interpreter = int_gestos
        clases_actuales = clases_gestos
        cooldown_tiempo = 1.0  # Delay para navegación

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if res.multi_hand_landmarks:
        for hand_lms in res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # --- PREDICCIÓN BASADA EN LANDMARKS (Coordenadas) ---
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
            
            prob_sena = np.max(out[0])
            if prob_sena > 0.80:
                idx = np.argmax(out[0])
                nombre_sena = clases_actuales[idx]
            
            # Variables para Mouse
            x_indice = int(hand_lms.landmark[8].x * w)
            y_indice = int(hand_lms.landmark[8].y * h)
            x_pulgar = int(hand_lms.landmark[4].x * w)
            y_pulgar = int(hand_lms.landmark[4].y * h)

    # Lógica de tiempos
    tiempo_actual = time.time()
    puede_ejecutar = (tiempo_actual - ultimo_comando) > cooldown_tiempo

    # UI: HUD Principal
    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(frame, f"ESTADO: {estado} | SENA: {nombre_sena}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- MÁQUINA DE ESTADOS ---
    if estado == 'MENU_PRINCIPAL':
        cv2.putText(frame, "Posa 'MODO_CONFIG' o 'MODO_PANTALLA'", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if puede_ejecutar:
            if nombre_sena == "MODO_CONFIG":
                estado = 'CONFIGURACION'
                ultimo_comando = tiempo_actual
            elif nombre_sena == "MODO_PANTALLA":
                estado = 'PANTALLA'
                ultimo_comando = tiempo_actual

    elif estado == 'CONFIGURACION':
        cv2.putText(frame, "BRILLO / VOLUMEN | 'MENU' para salir", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
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
                pyautogui.press('volumeup', presses=3) 
                ultimo_comando = tiempo_actual
            elif nombre_sena == "BAJAR_VOLUMEN":
                pyautogui.press('volumedown', presses=3)
                ultimo_comando = tiempo_actual
            elif nombre_sena == "MUTE":
                pyautogui.press('volumemute')
                ultimo_comando = tiempo_actual
            elif nombre_sena == "SCREENSHOT":
                pyautogui.press('printscreen')
                cv2.putText(frame, "CAPTURA!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                ultimo_comando = tiempo_actual
            elif nombre_sena == "MENU":
                estado = 'MENU_PRINCIPAL'
                ultimo_comando = tiempo_actual

    elif estado == 'PANTALLA':
        cv2.putText(frame, "MOUSE ACTIVO | Juntar dedos: CLICK", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
        if res.multi_hand_landmarks:
            mouse_x = np.interp(x_indice, [0, w], [0, screen_w])
            mouse_y = np.interp(y_indice, [0, h], [0, screen_h])
            pyautogui.moveTo(mouse_x, mouse_y, duration=0.1)
            
            # Click por cercanía de dedos
            if math.hypot(x_indice - x_pulgar, y_indice - y_pulgar) < 30:
                if (tiempo_actual - ultimo_comando) > 0.6: 
                    pyautogui.click()
                    cv2.circle(frame, (x_indice, y_indice), 20, (0, 0, 255), -1)
                    ultimo_comando = tiempo_actual

        if puede_ejecutar:
            if nombre_sena == "TECLADO":
                estado = 'TECLADO'
                ultimo_comando = tiempo_actual 
            elif nombre_sena == "MENU":
                estado = 'MENU_PRINCIPAL'
                ultimo_comando = tiempo_actual

    elif estado == 'TECLADO':
        cv2.putText(frame, "MODO ESCRITURA | 'OK' para salir", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
        if puede_ejecutar and nombre_sena not in ["NEUTRAL"]:
            if nombre_sena == "OK":
                estado = 'PANTALLA'
            elif nombre_sena == "ESPACIO":
                pyautogui.press('space')
            elif nombre_sena == "BORRAR":
                pyautogui.press('backspace')
            else:
                pyautogui.write(nombre_sena.lower())
            ultimo_comando = tiempo_actual

    # Barra de Cooldown (Visual)
    if not puede_ejecutar:
        pct = (tiempo_actual - ultimo_comando) / cooldown_tiempo
        cv2.rectangle(frame, (0, h-10), (int(w*pct), h), (0, 0, 255), -1)

    cv2.imshow("Control PC por Gestos v2.0", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()