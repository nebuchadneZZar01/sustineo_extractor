import camelot
import os

class TableToCSV:
    def __init__(self, path = str):
        self.__path = path
        self.__filename = os.path.basename(self.__path)[:-4]

        self.__tables = camelot.read_pdf(self.__path, flavor='lattice', pages='1-end')
        self.__out_path = os.path.join(os.getcwd(), 'out', 'table')

    @property
    def filename(self):
        return self.__filename

    @property
    def out_path(self):
        return self.__out_path

    def run(self):
        if not os.path.isdir(self.out_path):
            os.mkdir(self.out_path)

        output_dir = os.path.join(self.out_path, self.filename)

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        
        n_table = 1
        for i, table in enumerate(self.__tables):
            df = table.df
            page = table.page
            prev_page = self.__tables[i - 1].page if i > 0 else page

            # if there is more than one table in the same page,
            # then increase the counter so to have the "index"
            # of that table for each page
            if prev_page == page and i > 0:
                n_table += 1
            else:
                n_table = 1

            print(f'Table {n_table} at page {page}')
            print(df.head)
            print('\n')

            # exports the table into a csv file
            csv_filename = f'{self.filename}_p{page}_{n_table}.csv'
            table.to_csv(os.path.join(output_dir, csv_filename))