import cv2 as cv
import numpy as np
import math
from keras.models import load_model

from utils.letters import LETTERS

class CharacterRecognizer:
    def __init__(self, model_path):
        try:
            self.model = load_model(model_path)
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            self.model = None
        self.letters = LETTERS

    @staticmethod
    def refine_image(image):
        """Refina la imagen para asegurar que el tamaño y el padding sean correctos."""
        PADDING_SIZE, TARGET_SIZE = 22, 28
        rows, cols = image.shape
        
        # Determinar el factor de escala
        factor = PADDING_SIZE / max(rows, cols)
        rows, cols = int(round(rows * factor)), int(round(cols * factor))
        
        # Redimensionar la imagen
        resized_image = cv.resize(image, (cols, rows))
        
        cols_padding = (int(math.ceil((TARGET_SIZE - cols) / 2.0)), int(math.floor((TARGET_SIZE - cols) / 2.0)))
        rows_padding = (int(math.ceil((TARGET_SIZE - rows) / 2.0)), int(math.floor((TARGET_SIZE - rows) / 2.0)))
        
        padded_image = np.pad(resized_image, (rows_padding, cols_padding), 'constant')
        return padded_image
    
    def extract_and_refine(self, image, contour):
        """Extrae el contorno de la imagen y lo refina."""
        x, y, w, h = cv.boundingRect(contour)

        # si el contorno es muy pequeño, ignorarlo
        if w < 10 or h < 40:
            return None
        roi = cv.bitwise_not(image[y:y+h, x:x+w])
        padded_img = self.refine_image(roi)
        refined_image = cv.transpose(padded_img)
        return refined_image

    def predict_character(self, image):
        """Realiza la predicción de un carácter usando el modelo cargado."""
        if self.model is None:
            return None
        image = image.reshape(-1, 28, 28, 1)
        prediction = np.argmax(self.model.predict(image))
        return self.letters.get(prediction, None)

    def process_image(self, path):
        """Procesa una imagen completa y devuelve la cadena de caracteres reconocidos."""
        image = cv.imread(path, cv.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error al leer la imagen: {path}")
            return "No words found"
        _, thresh = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        contours_index = sorted(enumerate(contours), key=lambda c: cv.boundingRect(c[1])[0])

        predicted_chars = []
        for index, contour in contours_index:
            if hierarchy[0][index][3] != -1: 
                refined_image = self.extract_and_refine(image, contour)  

                if refined_image is None:
                    continue
                predicted_char = self.predict_character(refined_image)
                predicted_chars.append(predicted_char)

        return ''.join(filter(None, predicted_chars)) or "No words found"
