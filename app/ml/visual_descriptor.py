from PIL import Image
import numpy as np
import cv2


class VisualDescriptorExtractor:

    def extract(self, image: Image.Image) -> dict:
        rgb = np.array(image)

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        brightness = self._brightness(gray)
        contrast = self._contrast(gray)
        saturation = self._saturation(hsv)
        warmth = self._warmth(rgb)

        histogram = self._color_histogram(rgb)

        return {
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "warmth": warmth,
            "color_histogram": histogram,
        }

    def _brightness(self, gray):
        return float(np.mean(gray) / 255.0)

    def _contrast(self, gray):
        return float(np.std(gray) / 255.0)

    def _saturation(self, hsv):
        return float(np.mean(hsv[:, :, 1]) / 255.0)

    def _warmth(self, rgb):
        red_mean = np.mean(rgb[:, :, 0])
        blue_mean = np.mean(rgb[:, :, 2])

        warmth = (red_mean - blue_mean + 255) / 510
        return float(warmth)

    def _color_histogram(self, rgb):
        histogram = []

        for channel in range(3):
            hist = cv2.calcHist(
                [rgb],
                [channel],
                None,
                [16],
                [0, 256]
            )

            hist = cv2.normalize(hist, hist).flatten()
            histogram.extend(hist.tolist())

        return histogram