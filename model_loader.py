import pickle
import base64
import cv2
import numpy as np
import mediapipe as mp

class HandSignPredictor:
    def __init__(self, model_path):
        # Cargar el modelo entrenado
        model_dict = pickle.load(open(model_path, "rb"))
        self.model = model_dict["model"]

        # Inicializar MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

        # Diccionario de etiquetas
        self.labels_dict = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
            8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
            15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
            22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2',
            29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9',
            36: 'SI', 37: 'NO', 38: 'GRACIAS', 39: 'I LOVE YOU', 40: 'HELLO'
        }

    def decode_base64_image(self, image_base64):
        # Decodificar imagen base64
        header, encoded = image_base64.split(",", 1)
        image_data = base64.b64decode(encoded)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def preprocess_image(self, img):
        # Procesar imagen con MediaPipe para obtener landmarks
        H, W, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = []
                y_ = []
                data_aux = []

                for i in range(21):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(21):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                if len(data_aux) == 42:
                    return np.asarray(data_aux).reshape(1, -1)

        return None

    def predict(self, image_base64):
        try:
            img = self.decode_base64_image(image_base64)
            processed_data = self.preprocess_image(img)

            if processed_data is not None:
                prediction = self.model.predict(processed_data)
                return self.labels_dict[int(prediction[0])]
            else:
                return "No se detectaron manos"
        except Exception as e:
            print("Error al predecir:", e)
            return "Error"


# import random

# def predict_from_base64(image_base64: str) -> str:
#     return random.choice("ABCDEFGHIJKLMNÑOPQRSTUVWXYZ")




# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import io
# import base64

# # Cargar modelo entrenado una vez
# model = tf.keras.models.load_model("modelo.h5")  # cambia por tu ruta

# # Etiquetas (ajusta a tu modelo)
# CLASSES = list("ABCDEFGHIJKLMNÑOPQRSTUVWXYZ")

# def predict_from_base64(base64_str):
#     # Quitar prefijo "data:image/jpeg;base64,..."
#     header, encoded = base64_str.split(',', 1)
#     img_data = base64.b64decode(encoded)
#     image = Image.open(io.BytesIO(img_data)).convert("RGB")
#     print(image.size)
#     # Preprocesar según tu modelo
#     image = image.resize((64, 64))  # tamaño del modelo
#     image = np.array(image) / 255.0
#     image = np.expand_dims(image, axis=0)

#     prediction = model.predict(image)[0]
#     letra = CLASSES[np.argmax(prediction)]

#     return letra
