import os
import argparse

from lib.pdf2plot.plot_extractor import PDFToImage
from lib.pdf2plot.tables_extractor import TableToCSV

def main(pdf_path, language, debug, size_factor):
    print(f'File {pdf_path} selected')
    plot_extr = PDFToImage(pdf_path, language, debug, size_factor)
    plot_extr.run()
    table_extr = TableToCSV(pdf_path)
    table_extr.run()

    filename = os.path.basename(pdf_path)[:-4]
    out_plot_path = os.path.join('out', filename, 'img', 'plot')
    out_matrix_path = os.path.join('out', filename, 'img', 'm_matrix')
    out_table_path = os.path.join('out', filename, 'table')

    cnt_plots = 0
    cnt_matrices = 0
    cnt_tables = 0

    if os.path.isdir(out_plot_path):
        cnt_plots = len(os.listdir(out_plot_path))

    if os.path.isdir(out_matrix_path):
        cnt_matrices = len(os.listdir(out_matrix_path))
    
    if os.path.isdir(out_table_path):
        cnt_tables = len(os.listdir(out_table_path))

    print(f'{cnt_plots} plots were extracted in {out_plot_path}')
    print(f'{cnt_matrices} materiality matrices were extracted in {out_matrix_path}')
    print(f'{cnt_tables} tables were extracted in {out_table_path}\n')

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

    parser.add_argument('-s', '--size-factor', type=float, default=3,\
                        help='if used in debug mode, the image sizes will be divided by the choosen\
                            scale factor for a better visualization on lower resolution screens (default=3)')
    
    args = parser.parse_args()

    if os.path.isfile(args.pathname):
        if args.pathname[:-3].lower != 'pdf':
            print(f'{args.pathname} is not a PDF file\nPlease choose a PDF file')
            exit()
        else:
            main(args.pathname, args.language, args.debug_mode, args.size_factor)
    else:
        if os.path.isdir(args.pathname):
            n_files = len(os.listdir(args.pathname))
            for i, fn in enumerate(os.listdir(args.pathname)):
                complete_fn = os.path.join(args.pathname, fn)
                print(f'Extracting data from file {i+1} of {n_files}...\n')
                main(complete_fn, args.language, args.debug_mode, args.size_factor)
        else:
            print(f'ERROR: File {args.pathname} does not exist')