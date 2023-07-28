import os
import argparse

from lib.pdf2plot.plot_extractor import PDFToImage
from lib.pdf2plot.tables_extractor import TableToCSV

def main(pdf_path, language, debug, size_factor):
    print('Loading pdf...')
    plot_extr = PDFToImage(pdf_path, language, debug, size_factor)
    table_extr = TableToCSV(pdf_path)
    print('--- SEARCHING FOR MATERIALITY MATRICES ---\n')
    plot_extr.run()
    print('\n--- SEARCHING FOR TABLES ---\n')
    table_extr.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='sustineo_pdf2plot',
                                description='This program extracts materiality matrices in raster-format and tables in csv format from pdf files.\
                                            \nAuthor: nebuchadneZZar01 (Michele Ferro)\
                                            \nGitHub: https://github.com/nebuchadneZZar01/',\
                                            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('pathname')

    parser.add_argument('-l', '--language', type=str, default='ita',\
                        help='language of the plot to extract (default="ita")')

    parser.add_argument('-d', '--debug-mode', action='store_true',\
                        help='activate the visualization of the various passes')

    parser.add_argument('-s', '--size-factor', type=float, default=1.5,\
                        help='if used in debug mode, the image sizes will be divided by the choosen\
                            scale factor for a better visualization on lower resolution screens (default=1.5)')
    
    args = parser.parse_args()

    if os.path.isfile(args.pathname):
        main(args.pathname, args.language, args.debug_mode, args.size_factor)
    else:
        if os.path.isdir(args.pathname):
            n_files = len(os.listdir(args.pathname))
            for i, fn in enumerate(os.listdir(args.pathname)):
                complete_fn = os.path.join(args.pathname, fn)
                print('Converting file {n} of {n_files}...\n'.format(n = i+1, n_files = n_files))
                main(complete_fn, args.language, args.debug_mode, args.size_factor)
        else:
            print('ERROR: File {fn} does not exist'.format(fn = args.pathname))