import os
import pdfplumber
import pandas as pd
import re

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
        # gri values file
        self.__out_gri_values = os.path.join(self.__out_path, 'gri_values.txt')

        # extraction stats
        self.ex_table_other_cnt = 0
        self.ex_table_gri_cnt = 0

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
    
    @property
    def out_gri_values(self):
        return self.__out_gri_values
    
    def extract_gri_themes(self, table_df: pd.DataFrame):
        for i, row in table_df.iterrows():
            for j in range(len(row)):
                if row[j] is not None:
                    match_re = re.search('(.*?)(.(\s(\D|\S)\s)|\S?)(.\d?)', row[j])
                
                    if match_re is not None:
                        with open(self.out_gri_values, 'a') as f:
                            f.write(f'{row.values}\n\n')
                    else:
                        continue
    
    def run(self):
        pbar = tqdm(self.pdf_doc.pages)
        pbar.set_description('Extracting tables')

        with open(self.out_gri_values, 'w') as f:
            f.write(f'GRI values found in {self.filename}.pdf:\n')

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
                        self.ex_table_gri_cnt += 1
                        if not os.path.isdir(self.out_gri_table_path):
                            os.makedirs(self.out_gri_table_path)
                        df.to_csv(os.path.join(self.out_gri_table_path, csv_filename))
                        
                        self.extract_gri_themes(df)
                    else:
                        self.ex_table_other_cnt += 1
                        if not os.path.isdir(self.out_other_table_path):
                            os.makedirs(self.out_other_table_path)
                        df.to_csv(os.path.join(self.out_other_table_path, csv_filename))
    
    # print table extraction stats on cli
    def get_stats(self):
        total_amount = self.ex_table_gri_cnt + self.ex_table_other_cnt 

        print(f'{total_amount} table extractions were made:')
        print(f'- {self.ex_table_gri_cnt} gri tables were extracted in {self.out_gri_table_path}')
        print(f'- {self.ex_table_other_cnt} other tables were extracted in {self.out_other_table_path}')

    # output table extraction stats on a logfile   
    def file_stats(self, file_output: any):
        total_amount = self.ex_table_gri_cnt + self.ex_table_other_cnt 

        file_output.write(f'{total_amount} table extractions were made:\n')
        file_output.write(f'- {self.ex_table_gri_cnt} gri tables were extracted in {self.out_gri_table_path}\n')
        file_output.write(f'- {self.ex_table_other_cnt} other tables were extracted in {self.out_other_table_path}\n')