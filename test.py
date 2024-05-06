import cv2
import numpy as np
import math
from keras.models import load_model

LETTERS = {
     0: 'A',  1: 'B',  2: 'C',  3: 'D',  4: 'E',  5: 'F',  6: 'G',  7: 'H',  8: 'I',  9: 'J', 
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

class CharacterRecognizer:
    def __init__(self, model_path):
        # Cargar el modelo de red neuronal previamente entrenado
        self.model = load_model(model_path)

    @staticmethod
    def refine_image(gray):
        """Refina la imagen para asegurar que el tamaño y el padding sean correctos."""
        padding_size, target_size = 22, 28
        rows, cols = gray.shape
        
        # Determinar el factor de escala
        factor = padding_size / max(rows, cols)
        rows, cols = int(round(rows * factor)), int(round(cols * factor))
        
        # Redimensionar la imagen
        gray = cv2.resize(gray, (cols, rows))
        
        # Calcular el padding necesario
        cols_padding = (int(math.ceil((target_size - cols) / 2.0)), int(math.floor((target_size - cols) / 2.0)))
        rows_padding = (int(math.ceil((target_size - rows) / 2.0)), int(math.floor((target_size - rows) / 2.0)))
        
        # Aplicar padding
        gray = np.lib.pad(gray, (rows_padding, cols_padding), 'constant')
        return gray
    
    def extract_and_refine(self, image, contour):
        """Extrae el contorno de la imagen y lo refina."""
        x, y, w, h = cv2.boundingRect(contour)

        # si el contorno es muy pequeño, ignorarlo
        if w < 10 or h < 10:
            return None
        roi = cv2.bitwise_not(image[y:y+h, x:x+w])
        padded_img = self.refine_image(roi)
        refined_image = cv2.transpose(padded_img)
        return refined_image

    def predict_character(self, image):
        """Realiza la predicción de un carácter usando el modelo cargado."""
        image = image.reshape(1, 28, 28, 1)
        pred = np.argmax(self.model.predict(image))
        return LETTERS.get(pred, None)

    def process_image(self, path):
        """Procesa una imagen completa y devuelve la cadena de caracteres reconocidos."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contours_index = sorted(enumerate(contours), key=lambda x: cv2.boundingRect(x[1])[0])

        predicted_chars = []
        for index, contour in contours_index:
            if hierarchy[0][index][3] != -1: 
                refined_image = self.extract_and_refine(img, contour)  

                if refined_image is None:
                    continue
                predicted_char = self.predict_character(refined_image)
                predicted_chars.append(predicted_char)
                print(predicted_char)
                cv2.imshow('Character', refined_image)
                cv2.waitKey(0)

        cv2.destroyAllWindows()
        return ''.join(filter(None, predicted_chars)) or "No hay palabra"

recognizer = CharacterRecognizer('models/saved_models/model.keras')
print(f'La predicción es: {recognizer.process_image("test/z.png")}')

