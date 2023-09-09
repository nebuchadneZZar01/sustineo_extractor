import os
import pdfplumber
import pandas as pd

from tqdm import tqdm

class TableToCSV:
    """Object for extracting tables into CSV-format files from PDF-format files.
    
    Keyword Arguments:
        - path -- Path to the PDF file to extract
    """
    def __init__(self, path: str):
        self.__path = path
        self.__filename = os.path.basename(self.__path)[:-4]

        self.__pdf_doc = pdfplumber.open(self.__path)

        self.__out_path = os.path.join(os.getcwd(), 'out', self.__filename)

        # gri tables location
        self.__out_gri_table_path = os.path.join(self.__out_path, 'table', 'gri')
        # other type tables location
        self.__out_other_table_path = os.path.join(self.__out_path, 'table', 'other')

    @property
    def filename(self):
        return self.__filename

    @property
    def pdf_doc(self):
        return self.__pdf_doc

    @property
    def out_gri_table_path(self):
        return self.__out_gri_table_path

    @property
    def out_other_table_path(self):
        return self.__out_other_table_path
    
    def run(self):
        pbar = tqdm(self.pdf_doc.pages)
        pbar.set_description('Extracting tables')
        for page in pbar:
            tables = page.find_tables()                 # list of tables in the page
            text = page.extract_text()   
            page_number = page.page_number              # page number of pdf actual page

            for i, table in enumerate(tables):
                df = pd.DataFrame.from_dict(table.extract())

                if not df.empty:
                    # exports the table into a csv file
                    csv_filename = f'page_{page_number}-{i+1}.csv'

                    if 'gri' in text.lower():
                        if not os.path.isdir(self.out_gri_table_path):
                            os.makedirs(self.out_gri_table_path)
                        df.to_csv(os.path.join(self.out_gri_table_path, csv_filename))
                    else:
                        if not os.path.isdir(self.out_other_table_path):
                            os.makedirs(self.out_other_table_path)
                        df.to_csv(os.path.join(self.out_other_table_path, csv_filename))