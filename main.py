import cv2
from ocr import OCR

import fractions
from template_matching import Matcher 

def main(image_dir, template_dir, scale_visualization):
    image = cv2.imread(image_dir)
    template = cv2.imread(template_dir)
    image_size = image.shape
    new_size = (int(image_size[1]/scale_visualization), int(image_size[0]/scale_visualization))
    image_scaled = cv2.resize(image, new_size)
    cv2.imshow('original', image_scaled)
    cv2.waitKey(0)

    m = Matcher(template, image)
    plot, label = m.separate_image()

    ocr = OCR(plot, scale_visualization)
    ocr.process_text()

    ocr.show_image()

im = 'amadori2.png'
tmp = 'template.png'
main(im, tmp, 1.5)