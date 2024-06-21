import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='./dataset', help='name of folder')
parser.add_argument('--source', type=str, default='./archive', help='path to data')
parser.add_argument('--train', type=float, default=0.7, help='decimal of train data')
parser.add_argument('--test', type=float, default=0.1, help='decimal of test data')
args = parser.parse_args()

OUTPUT_FOLDER = args.name
SOURCE_FOLDER = args.source
TRAIN_SIZE = args.train
TEST_SIZE = args.test

def create_data_folders(folder_name : str = OUTPUT_FOLDER) -> None:
    """Creates a new folder where splitted data will be saved"""

    if os.path.exists(folder_name):
        _output_folder = folder_name + '(Copy)'
        os.mkdir(f'{_output_folder} (Copy)')
    else:
        _output_folder = folder_name
        os.mkdir(folder_name)
        
    print(f"Created {_output_folder} directory.")
    
    for folder in ['images', 'labels']:
        for i in ['train', 'test', 'val']:
            os.makedirs(_output_folder + '/' + folder + '/' + i)


def move_images(folder_name : str = OUTPUT_FOLDER):
    """Moves images to created directory"""
    number_of_picture = len(list(os.listdir(folder_name)))
    print(number_of_picture)
    
    
if __name__ == '__main__':
    create_data_folders()