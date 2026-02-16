
import cv2
import numpy as np
import tensorflow as tf

# Cargar el modelo TFLite
interpreter = tf.lite.Interpreter(model_path="model_unquant2.tflite")
interpreter.allocate_tensors()

# Obtener detalles de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = [line.strip() for line in open("labels2.txt", "r").readlines()]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    
    # Preprocesar imagen para TFLite (224x224)
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    img = (img / 127.5) - 1 # Normalizar

    # Ejecutar predicción
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    index = np.argmax(output_data[0])
    probabilidad = output_data[0][index]

    if probabilidad > 0.85:
        nombre_sucio = class_names[index]
        nombre_limpio = "".join([i for i in nombre_sucio if not i.isdigit()]).strip()
        
        texto = f"{nombre_limpio}"
        cv2.putText(frame, texto, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Detector de Señas (ABECEDARIO)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()