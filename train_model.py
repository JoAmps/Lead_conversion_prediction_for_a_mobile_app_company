import logging
from functions.data_preprocess import process_data
from load_data import load_data

logging.basicConfig(
    filename='./outputs/process.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    pass    