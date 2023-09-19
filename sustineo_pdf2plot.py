import os
import argparse

from lib.pdf2plot.plot_extractor import PDFToImage
from lib.pdf2plot.tables_extractor import TableToCSV

def main(pdf_path, language, headless, user_correction, paragraph, dataset_creation, debug, size_factor):
    print(f'File {pdf_path} selected')
    plot_extr = PDFToImage(pdf_path, language, headless, user_correction, paragraph, dataset_creation, debug, size_factor)
    plot_extr.run()

    if not dataset_creation:
        table_extr = TableToCSV(pdf_path)
        table_extr.run()

    print(f'{os.path.basename(pdf_path)} extraction report:')
    plot_extr.get_stats()

    if not dataset_creation:
        table_extr.get_stats()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='sustineo_pdf2plot',
                                    description='This program extracts materiality matrices in raster-format and tables in csv format from pdf files.\
                                                \nAuthor: nebuchadneZZar01 (Michele Ferro)\
                                                \nGitHub: https://github.com/nebuchadneZZar01/',\
                                                formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('pathname')

    parser.add_argument('-l', '--language', type=str, default='ita',\
                        help='language of the plot to extract (default="ita")')
    
    parser.add_argument('-hm', '--headless', action='store_true',\
                        help='run in headless mode in lack of GUI environment')
    
    parser.add_argument('-c', '--correction', action='store_true',\
                        help='enable user correction')
    
    parser.add_argument('-p', '--paragraph', action='store_true',\
                        help='enable automatic paragraph removal (new images will be saved in <file>/img/plot/no_par)')
    
    parser.add_argument('-dc', '--dataset-creation', action='store_true',\
                        help='enable dataset creation mode')
    
    parser.add_argument('-d', '--debug-mode', action='store_true',\
                        help='enable the visualization of the various passes')
    
    parser.add_argument('-s', '--size-factor', type=float, default=3,\
                        help='if used in debug mode, the image sizes will be divided by the choosen\
                            scale factor for a better visualization on lower resolution screens (default=3)')
    
    args = parser.parse_args()

    if os.path.isfile(args.pathname):
        if args.pathname.endswith('pdf'):
            main(args.pathname, args.language, args.headless, args.correction, args.paragraph, args.dataset_creation, args.debug_mode, args.size_factor)
        else:
            print(f'{args.pathname} is not a PDF file\nPlease choose a PDF file')
            exit()
    else:
        if os.path.isdir(args.pathname):
            n_files = len(os.listdir(args.pathname))
            for i, fn in enumerate(os.listdir(args.pathname)):
                complete_fn = os.path.join(args.pathname, fn)
                print(f'Extracting data from file {i+1} of {n_files}...\n')
                main(complete_fn, args.language, args.headless, args.correction, args.paragraph, args.dataset_creation, args.debug_mode, args.size_factor)
                os.remove(complete_fn)
        else:
            print(f'ERROR: File {args.pathname} does not exist')