import os
import pdfplumber
import pandas as pd

from tqdm import tqdm

class TableToCSV:
    def __init__(self, path = str):
        self.__path = path
        self.__filename = os.path.basename(self.__path)[:-4]

        self.__pdf_doc = pdfplumber.open(self.__path)

        self.__out_path = os.path.join(os.getcwd(), 'out', self.__filename)
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
        if not os.path.isdir(self.out_table_path):
            os.makedirs(self.out_table_path)

        pbar = tqdm(self.pdf_doc.pages)
        pbar.set_description('Extracting tables')
        for page in pbar:
            table = page.extract_table()                # table extracted from pdf actual page        
            page_number = page.page_number              # page number of pdf actual page
            
            df = pd.DataFrame.from_dict(table)

            if not df.empty:
                # exports the table into a csv file
                csv_filename = f'page_{page_number}.csv'
                df.to_csv(os.path.join(self.out_table_path, csv_filename))