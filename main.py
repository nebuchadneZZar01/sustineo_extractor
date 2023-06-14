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

    print('--- START PLOT LOG ---')
    if plot_type == 'blob':
        ocr = PlotOCR_Blob(plot, language, scale_visualization, debug_mode)
    elif plot_type == 'box':
        ocr = PlotOCR_Box(plot, language, scale_visualization, debug_mode)
    ocr.process_image()

    labelboxes = ocr.get_data()
    plot_col = ocr.get_colors_hsv()
    if debug_mode: ocr.show_image()
    print('--- STOP PLOT LOG ---')
    
    print('\n--- START LEGEND LOG ---')
    if legend is not None:

        leg = LegendOCR(legend, labelboxes, language, scale_visualization, debug_mode)
        leg.process_image(plot_col)
        legendboxes = leg.get_data()
        if debug_mode: leg.show_image()
    else:
        legendboxes = None
        print('There is no legend in this plot, skipping legend elaboration...')
    print('--- STOP LEGEND LOG ---\n')

    ex = Exporter(image_dir, labelboxes, legendboxes)
    ex.compose_export_dataset()
    ex.compose_export_plot()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='sustineo_extractor',
                                description='This program extracts data from materiality matrices and reinterprets them in a more undestandable form.\
                                            \nAuthor: nebuchadneZZar01 (Michele Ferro)\
                                            \nGitHub: https://github.com/nebuchadneZZar01/',\
                                            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('pathname')

    parser.add_argument('-l', '--language', type=str, default='ita',\
                        help='language of the plot to extract (default="ita")')

    parser.add_argument('-t', '--type', type=str, default='box',\
                        help='type of plot from where extract the informations (possible types=["box", "blob"], default="box")')

    parser.add_argument('-d', '--debug-mode', action='store_true',\
                        help='activate the visualization of the various passes')

    parser.add_argument('-s', '--size-factor', type=float, default=1.5,\
                        help='if used in debug mode, the image sizes will be divided by the choosen\
                            scale factor for a better visualization on lower resolution screens (default=1.5)')

    args = parser.parse_args()

    if os.path.isfile(args.pathname):
        # takes in input file
        main(args.pathname, args.language, args.type, args.size_factor, args.debug_mode)
        print('Extraction completed!')
    else:
        # takes in input entire directory
        if os.path.isdir(args.pathname):
            n_files = len(os.listdir(args.pathname))
            for i, fn in enumerate(os.listdir(args.pathname)):
                complete_fn = os.path.join(args.pathname, fn)
                print('Extracting file {n} of {n_files}...\n'.format(n = i+1, n_files = n_files))
                try:
                    main(complete_fn, args.language, args.type, args.size_factor, args.debug_mode)
                except:
                    print('There was an error extracting data from file {fn}!'.format(fn = args.pathname))
                    continue
            print('Extraction completed!')
        else:
            print('ERROR: File {fn} does not exist'.format(fn = args.filename))