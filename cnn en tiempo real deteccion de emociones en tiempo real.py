import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Define las emociones y las teclas para capturarlas
EMOCIONES = {
    "f": "feliz",
    "t": "triste",
    "e": "enojado",
    "n": "neutral",
    "s": "sorprendido"
}

IMG_SIZE = 48
imagenes = []
etiquetas = []

print("Instrucciones:")
print("- Mira a la cámara y presiona una tecla para capturar tu emoción:")
for k, v in EMOCIONES.items():
    print(f"    '{k}' para '{v}'")
print("- Cuando termines de capturar, presiona 'q' para entrenar el modelo.")
print("- Luego el sistema reconocerá tu emoción en tiempo real.")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    min_dim = min(h, w)
    x1 = w//2 - min_dim//4
    y1 = h//2 - min_dim//4
    x2 = w//2 + min_dim//4
    y2 = h//2 + min_dim//4
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(frame, "Presiona una tecla para etiquetar la emocion", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    cv2.imshow("Captura emociones", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    for tecla, emocion in EMOCIONES.items():
        if key == ord(tecla):
            cara = frame[y1:y2, x1:x2]
            cara = cv2.cvtColor(cara, cv2.COLOR_BGR2GRAY)
            cara = cv2.resize(cara, (IMG_SIZE, IMG_SIZE))
            imagenes.append(cara)
            etiquetas.append(emocion)
            print(f"Capturada emocion: {emocion} (total: {len(imagenes)})")

cap.release()
cv2.destroyAllWindows()

if len(imagenes) < len(EMOCIONES) * 3:
    print("¡Debes capturar más imágenes de cada emoción para entrenar!")
    exit()

# Prepara datos
imagenes = np.array(imagenes).astype("float32") / 255.0
imagenes = np.expand_dims(imagenes, -1)
emocion_a_idx = {e:i for i, e in enumerate(sorted(set(etiquetas)))}
idx_a_emocion = {i:e for e,i in emocion_a_idx.items()}
y = np.array([emocion_a_idx[e] for e in etiquetas])
y = tf.keras.utils.to_categorical(y, num_classes=len(emocion_a_idx))

# Modelo simple CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(emocion_a_idx), activation='softmax')
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("Entrenando modelo...")

model.fit(imagenes, y, epochs=15, batch_size=8)

print("¡Entrenamiento terminado! Presiona ESC para salir en cualquier momento.")

# Webcam: predicción en tiempo real
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    min_dim = min(h, w)
    x1 = w//2 - min_dim//4
    y1 = h//2 - min_dim//4
    x2 = w//2 + min_dim//4
    y2 = h//2 + min_dim//4
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    cara = frame[y1:y2, x1:x2]
    cara = cv2.cvtColor(cara, cv2.COLOR_BGR2GRAY)
    cara = cv2.resize(cara, (IMG_SIZE, IMG_SIZE))
    cara = cara.astype("float32") / 255.0
    cara = np.expand_dims(cara, axis=(0,-1))

    pred = model.predict(cara, verbose=0)
    emocion_pred = idx_a_emocion[np.argmax(pred)]

    cv2.putText(frame, f"Emocion: {emocion_pred}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    cv2.imshow("Reconocimiento en tiempo real", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()