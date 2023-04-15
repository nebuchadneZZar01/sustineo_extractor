import argparse
import os
import cv2
from ocr import OCR
from plot_cropper import Cropper 

def main(image_dir, template_dir, scale_visualization, debug_mode):
    image = cv2.imread(image_dir)
    template = cv2.imread(template_dir)
    image_size = image.shape
    new_size = (int(image_size[1]/scale_visualization), int(image_size[0]/scale_visualization))
    image_scaled = cv2.resize(image, new_size)

    if debug_mode:
        cv2.imshow('original', image_scaled)
        cv2.waitKey(0)
    
    m = Cropper(template, image, debug_mode, scale_visualization)
    plot, label = m.separate_image()

    ocr = OCR(plot, image_dir, scale_visualization, debug_mode)
    ocr.process_text()
    ocr.extract_data()

    if debug_mode:
        ocr.show_image()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Sustineo Extractor',
                                description='This program extracts data from materiality matrices and reinterprets them in a more undestandable form.')

    parser.add_argument('filename')
    parser.add_argument('-s', '--size-factor', type=float, default=1.5)
    parser.add_argument('-d', '--debug-mode', type=bool, default=False)

    args = parser.parse_args()

    tmp = 'template.png'

    main(args.filename, tmp, args.size_factor, args.debug_mode)