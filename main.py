import cv2
from ocr import OCR

import fractions

def main(image_dir, scale_visualization):
    image = cv2.imread(image_dir)
    image_size = image.shape
    new_size = (int(image_size[1]/scale_visualization), int(image_size[0]/scale_visualization))
    image_scaled = cv2.resize(image, new_size)
    cv2.imshow('original', image_scaled)
    cv2.waitKey(0)

    ocr = OCR(image, scale_visualization)
    ocr.process_text()

    ocr.show_image()

im = 'amadori2.png'
main(im, 1.5)