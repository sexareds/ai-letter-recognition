import cv2
import numpy as np
import math
from keras.models import load_model

class CharacterPredictor:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.original_size = 20
        self.target_size = 28

    def refine_image(self, image):
        """Resize and pad the image to the target dimensions."""
        rows, cols = image.shape
        scaling_factor = self.original_size / max(rows, cols)
        rows, cols = int(round(rows * scaling_factor)), int(round(cols * scaling_factor))

        image = cv2.resize(image, (cols, rows))
        pad_width = (
            (math.ceil((self.target_size - rows) / 2), math.floor((self.target_size - rows) / 2)),
            (math.ceil((self.target_size - cols) / 2), math.floor((self.target_size - cols) / 2))
        )
        return np.pad(image, pad_width, mode='constant')

    def predict_character(self, refined_img):
        """Predict the character from the refined image."""
        prediction = self.model.predict(refined_img.reshape(1, 28, 28, 1))
        return chr(np.argmax(prediction) + 65)  # Convert prediction to character

    def process_image(self, image_path):
        """Process an image and predict characters."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Error: Image not found or cannot be loaded.")
            return "Image loading error"

        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Debugging: Visualize contours
        img_with_contours = img.copy()
        cv2.drawContours(img_with_contours, contours, -1, (0,255,0), 3)

        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])  # Sort by x

        predicted_chars = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            print(f"Contour at x:{x} y:{y} with width:{w} height:{h}")  # Debug output

            # Relaxing size criteria for testing
            if w > 8 and h > 8:  # Minimum size filter
                refined_img = self.extract_and_refine(img, contour)
                predicted_char = self.predict_character(refined_img)
                predicted_chars.append(predicted_char)
                print(f"Predicted character: {predicted_char}")  # Debug output
                
        # Show image with contours
        cv2.imshow('Contours', img_with_contours)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return ''.join(predicted_chars) if predicted_chars else "No prediction"

    def extract_and_refine(self, img, contour):
        """Extract and refine contour to prepare for prediction."""
        x, y, w, h = cv2.boundingRect(contour)
        cropped_img = img[y:y+h, x:x+w]
        refined_img = self.refine_image(cv2.bitwise_not(cropped_img))

        return refined_img

if __name__ == "__main__":
    predictor = CharacterPredictor('models/saved_models/model.keras')
    result = predictor.process_image('test/z.png')
    print(result)
