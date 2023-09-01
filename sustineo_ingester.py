import os
from elasticsearch import Elasticsearch, helpers
import configparser
import pandas as pd
import json

from tqdm import tqdm

config = configparser.ConfigParser()
config.read('config.ini')

out_dir = os.path.join(os.getcwd(), 'out')

def main():
    es = Elasticsearch(config['ELASTIC']['host'],
                    basic_auth=(config['ELASTIC']['user'], config['ELASTIC']['password']))

    pbar = tqdm(os.listdir(out_dir))
    pbar.set_description('Ingesting CSV data to Elasticsearch') 
    for dir in pbar:
        matrix_dir = os.path.join(out_dir, dir, 'csv')
        table_dir = os.path.join(out_dir, dir, 'table')
        
        # ingesting materiality matrix data
        if os.path.isdir(matrix_dir):
            for matrix in os.listdir(matrix_dir):
                df = pd.read_csv(os.path.join(matrix_dir, matrix))
                json_str = df.to_json(orient='records')

                # csv files cannot be ingested
                # we have to map in a json file
                # and then ingest it
                json_records = json.loads(json_str)
                action_list = []
                for row in json_records:
                    record = {
                        '_op_type': 'index',
                        '_index': 'sustineo_matrix',
                        '_source': row
                    }
                    action_list.append(record)
                
                helpers.bulk(es, action_list)

        # ingesting other tables data
        if os.path.isdir(table_dir):
            for table in os.listdir(table_dir):
                df = pd.read_csv(os.path.join(table_dir, table)) 
                json_str = df.to_json(orient='records')

                json_records = json.loads(json_str)
                action_list = []
                for row in json_records:
                    record = {
                        '_op_type': 'index',
                        '_index': 'sustineo_table',
                        '_source': row
                    }
                    action_list.append(record)
                try:
                    helpers.bulk(es, action_list)
                except:
                    # row not ingested
                    pass

if __name__ == '__main__':
    main()