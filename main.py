import argparse
import os
import cv2
from ocr import *
from plot_cropper import Cropper

def main(image_dir, language, scale_visualization, debug_mode):
    image = cv2.imread(image_dir)
    image_size = image.shape
    new_size = (int(image_size[1]/scale_visualization), int(image_size[0]/scale_visualization))
    image_scaled = cv2.resize(image, new_size)

    if debug_mode:
        cv2.imshow('original', image_scaled)
        cv2.waitKey(0)

    m = Cropper(image, debug_mode, scale_visualization)
    plot, legend = m.separate_image()

    print('--- PLOT LOG ---')

    ocr = PlotOCR(plot, image_dir, language, scale_visualization, debug_mode)
    ocr.process_text()

    labelboxes = ocr.extract_data()
    plot_col = ocr.get_colors_hsv()

    print('--- LEGEND LOG ---')

    leg = LegendOCR(legend, image_dir, labelboxes, language, scale_visualization, debug_mode)
    leg.process_text()
    leg.get_colors_position(plot_col)
    leg.process_legend()
    leg.construct_dataset()

    if debug_mode:
        ocr.show_image()
        leg.show_image()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='sustineo_extractor',
                                description='This program extracts data from materiality matrices and reinterprets them in a more undestandable form.\
                                            \nAuthor: nebuchadneZZar01 (Michele Ferro)\
                                            \nGitHub: https://github.com/nebuchadneZZar01/',\
                                            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('filename')
    parser.add_argument('-l', '--language', type=str, default='ita', help='language of the plot to extract (default="ita")')
    parser.add_argument('-d', '--debug-mode', type=bool, default=False, help='activate the visualization of the various passes (default=false)')
    parser.add_argument('-s', '--size-factor', type=float, default=1.5, help='if used in debug mode, the image sizes will be divided by the choosen scale factor for a better visualization on lower resolution screens (default=1.5)')

    args = parser.parse_args()

    main(args.filename, args.language, args.size_factor, args.debug_mode)