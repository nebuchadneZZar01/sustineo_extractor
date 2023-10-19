import os
import signal
import argparse

from lib.pdf2plot.plot_extractor import PDFToImage
from lib.pdf2plot.tables_extractor import TableToCSV

total_plot_w_ex = 0
total_plot_ex = 0
total_matrix_ex = 0
total_table_other_ex = 0
total_table_gri_ex = 0

def handler(signum, frame):
    res = input("Ctrl-C was pressed. Do you really want to exit? [Y/n] ")
    
    if res[0].lower() == 'y':
        print('Gracefully exiting the program...')

        print('Complexive extraction report:')
        print(f'- {total_matrix_ex} materiality matrix have been extracted;')
        print(f'- {total_plot_ex} plots have been extracted ({total_plot_w_ex} of which could require user intervention);')
        print(f'- {total_table_gri_ex} gri tables have been extracted;')
        print(f'- {total_table_other_ex} other tables have been extracted.')
        exit(0)

signal.signal(signal.SIGINT, handler)

def main(pdf_path, language, headless, user_correction, paragraph, dataset_creation, debug, size_factor):
    global total_plot_w_ex
    global total_plot_ex
    global total_matrix_ex
    global total_table_other_ex
    global total_table_gri_ex

    print(f'File {pdf_path} selected')
    plot_extr = PDFToImage(pdf_path, language, headless, user_correction, paragraph, dataset_creation, debug, size_factor)
    plot_extr.run()

    # complexive plot extraction stats
    total_plot_ex += plot_extr.ex_plot_cnt
    total_plot_w_ex += plot_extr.ex_plot_w_cnt
    total_matrix_ex += plot_extr.ex_materiality_mat_cnt

    if not dataset_creation:
        table_extr = TableToCSV(pdf_path)
        table_extr.run()

        # complexive table extraction stats
        total_table_gri_ex += table_extr.ex_table_gri_cnt
        total_table_other_ex += table_extr.ex_table_other_cnt

    print(f'{os.path.basename(pdf_path)} extraction report:')
    plot_extr.get_stats()

    if not dataset_creation:
        table_extr.get_stats()

    with open('logs.txt', 'a') as testf:
        testf.write(f'{os.path.basename(pdf_path)} extraction report:\n')
        plot_extr.file_stats(testf)

        if not dataset_creation:
            table_extr.file_stats(testf)
        
        testf.write('\n\n')


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
            main(args.pathname, args.language, 
                 args.headless, args.correction, args.paragraph, 
                 args.dataset_creation, args.debug_mode, args.size_factor)
        else:
            print(f'{args.pathname} is not a PDF file\nPlease choose a PDF file')
            exit()
    else:
        if os.path.isdir(args.pathname):
            n_files = len(os.listdir(args.pathname))
            for i, fn in enumerate(os.listdir(args.pathname)):
                complete_fn = os.path.join(args.pathname, fn)
                print(f'Extracting data from file {i+1} of {n_files}...\n')
                main(complete_fn, args.language, 
                     args.headless, args.correction, args.paragraph, 
                     args.dataset_creation, args.debug_mode, args.size_factor)
                
            print('Complexive extraction report:')
            print(f'- {total_matrix_ex} materiality matrix have been extracted;')
            print(f'- {total_plot_ex} plots have been extracted ({total_plot_w_ex} of which could require user intervention);')
            print(f'- {total_table_gri_ex} gri tables have been extracted;')
            print(f'- {total_table_other_ex} other tables have been extracted.')
        else:
            print(f'ERROR: File {args.pathname} does not exist')