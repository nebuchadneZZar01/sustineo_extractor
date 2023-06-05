import argparse
import os
import cv2
from ocr import *
from plot_cropper import Cropper, BlobCropper
from exporter import Exporter

def main(image_dir, language, plot_type, scale_visualization, debug_mode):
    image = cv2.imread(image_dir)
    
    if plot_type == 'box':
        m = Cropper(image, debug_mode, scale_visualization)
    elif plot_type == 'blob':
        m = BlobCropper(image, debug_mode, scale_visualization)
    plot, legend = m.separate_image()

    print('--- PLOT LOG ---')

    if plot_type == 'blob':
        ocr = PlotOCR_Blob(legend, language, scale_visualization, debug_mode)
    elif plot_type == 'box':
        ocr = PlotOCR_Box(plot, language, scale_visualization, debug_mode)
    ocr.process_image()

    labelboxes = ocr.get_data()
    plot_col = ocr.get_colors_hsv()
    if debug_mode: ocr.show_image()

    print('--- LEGEND LOG ---')

    leg = LegendOCR(legend, labelboxes, language, scale_visualization, debug_mode)
    leg.process_image(plot_col)
    legendboxes = leg.get_data()
    if debug_mode: leg.show_image()

    ex = Exporter(image_dir, labelboxes, legendboxes)
    ex.compose_export_dataset()
    ex.compose_export_plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='sustineo_extractor',
                                description='This program extracts data from materiality matrices and reinterprets them in a more undestandable form.\
                                            \nAuthor: nebuchadneZZar01 (Michele Ferro)\
                                            \nGitHub: https://github.com/nebuchadneZZar01/',\
                                            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('filename')
    parser.add_argument('-l', '--language', type=str, default='ita',\
                        help='language of the plot to extract (default="ita")')

    parser.add_argument('-t', '--type', type=str, default='box',\
                        help='type of plot from where extract the informations (default="box")')

    parser.add_argument('-d', '--debug-mode', action='store_true',\
                        help='activate the visualization of the various passes')

    parser.add_argument('-s', '--size-factor', type=float, default=1.5,\
                        help='if used in debug mode, the image sizes will be divided by the choosen\
                            scale factor for a better visualization on lower resolution screens (default=1.5)')

    args = parser.parse_args()

    if os.path.isfile(args.filename):
        main(args.filename, args.language, args.type, args.size_factor, args.debug_mode)
    else:
        print('ERROR: File {fn} does not exist'.format(fn = args.filename))