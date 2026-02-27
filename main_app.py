import os
import sys
import warnings
import platform
import numpy as np

# --- DETECCIÓN DE SISTEMA OPERATIVO ---
SISTEMA = platform.system()

# --- CONFIGURACIÓN PARA EVITAR ERRORES EN MAC (M1/M2/M3) ---
if SISTEMA == "Darwin":
    os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
    # En Mac el índice suele ser 0, pero si no abre prueba con 1
    INDICE_CAMARA = 0 
else:
    # En Windows el índice estándar es 0
    INDICE_CAMARA = 0

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

import cv2

# --- IMPORTACIÓN SEGURA DE TENSORFLOW ---
try:
    import tensorflow as tf
    # Acceso universal al Interpreter para evitar errores de ruta en Mac
    Interpreter = tf.lite.Interpreter
    print(f"✅ TensorFlow cargado en {SISTEMA}")
except Exception as e:
    print(f"❌ Error al cargar TensorFlow: {e}")
    sys.exit()

# --- IMPORTACIÓN SEGURA DE MEDIAPIPE ---
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    print("✅ MediaPipe cargado con éxito")
except Exception as e:
    print(f"❌ Error al cargar MediaPipe: {e}")
    sys.exit()

# --- CARGAR MODELO TFLITE ---
try:
    # Se usa la referencia 'Interpreter' definida arriba para máxima compatibilidad
    interpreter = Interpreter(model_path="model_unquant2.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    with open("labels2.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    print("✅ Modelo TFLite cargado")
except Exception as e:
    print(f"❌ Error con archivos del modelo: {e}")
    sys.exit()

# --- CONFIGURAR DETECTOR DE MANOS ---
# model_complexity=0 es vital para que no crashee en Mac y es más rápido en Windows
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# --- INICIALIZAR CÁMARA (LÓGICA HÍBRIDA) ---
if SISTEMA == "Darwin":
    # En Mac no se usa CAP_DSHOW
    cap = cv2.VideoCapture(INDICE_CAMARA)
else:
    # En Windows CAP_DSHOW es necesario para evitar lentitud al iniciar
    cap = cv2.VideoCapture(INDICE_CAMARA, cv2.CAP_DSHOW)

estado = 'menu'
letras_disponibles = ["A", "B", "C"]
indice_letra = 0

print(f"🚀 Iniciando aplicación en {SISTEMA}... Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: No se pudo acceder a la cámara.")
        break
    
    frame = cv2.flip(frame, 1)
    key = cv2.waitKey(1) & 0xFF

    if estado == 'menu':
        frame[:] = (40, 40, 40)
        cv2.putText(frame, "LSM PRO - UNIVERSAL", (70, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "1. APRENDER | 2. PRACTICAR", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if key == ord('1'): estado = 'aprender'
        if key == ord('2'): estado = 'practicar'

    elif estado == 'aprender':
        letra = letras_disponibles[indice_letra]
        img_guia = cv2.imread(f"imagenes_lsm/{letra}.jpg")
        if img_guia is not None:
            img_guia = cv2.resize(img_guia, (200, 200))
            frame[50:250, 50:250] = img_guia
        
        cv2.putText(frame, f"LETRA: {letra}", (280, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "A/D: Navegar | M: Menu", (280, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if key == ord('d'): indice_letra = (indice_letra + 1) % len(letras_disponibles)
        if key == ord('a'): indice_letra = (indice_letra - 1) % len(letras_disponibles)
        if key == ord('m'): estado = 'menu'

    elif estado == 'practicar':
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands_detector.process(rgb)

        if res.multi_hand_landmarks:
            for hand_lms in res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                
                # --- PREDICCIÓN ---
                try:
                    # Preprocesamiento de la imagen para el modelo
                    roi = cv2.resize(frame, (224, 224))
                    roi = np.expand_dims(roi, axis=0).astype(np.float32)
                    roi = (roi / 127.5) - 1
                    
                    interpreter.set_tensor(input_details[0]['index'], roi)
                    interpreter.invoke()
                    out = interpreter.get_tensor(output_details[0]['index'])
                    idx = np.argmax(out[0])
                    
                    # Limpiamos el nombre de la etiqueta
                    nombre = "".join([i for i in class_names[idx] if not i.isdigit()]).strip()
                    cv2.putText(frame, f"Detectado: {nombre}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                except Exception as e:
                    cv2.putText(frame, "Error prediccion", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Coloca tu mano frente a la camara", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if key == ord('m'): estado = 'menu'

    cv2.imshow("LSM Pro v3.3 - Multiplataforma", frame)
    if key == ord('q'): break

cap.release()
cv2.destroyAllWindows()