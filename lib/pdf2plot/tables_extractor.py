import os
import pdfplumber
import pandas as pd

class TableToCSV:
    def __init__(self, path = str):
        self.__path = path
        self.__filename = os.path.basename(self.__path)[:-4]

        self.__pdf_doc = pdfplumber.open(self.__path)

        self.__out_path = os.path.join(os.getcwd(), 'out')
        self.__out_table_path = os.path.join(self.out_path, 'table')

    @property
    def filename(self):
        return self.__filename

    @property
    def pdf_doc(self):
        return self.__pdf_doc

    @property
    def out_path(self):
        return self.__out_path

    @property
    def out_table_path(self):
        return self.__out_table_path

    def run(self):
        if not os.path.isdir(self.out_path):
            os.mkdir(self.out_path)

        if not os.path.isdir(self.out_table_path):
            os.mkdir(self.out_table_path)

        output_dir = os.path.join(self.out_table_path, self.filename)

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        
        for page in self.pdf_doc.pages:
            table = page.extract_table()                # table extracted from pdf actual page        
            page_number = page.page_number              # page number of pdf actual page
            
            df = pd.DataFrame.from_dict(table)

            if not df.empty:
                print(f'Table page {page_number}')
                print(df.head)
                print('\n')

                # exports the table into a csv file
                csv_filename = f'{self.filename}_p{page_number}.csv'
                df.to_csv(os.path.join(output_dir, csv_filename))