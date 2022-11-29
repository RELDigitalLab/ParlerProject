import argparse
import json
import os
import pandas as pd
from tqdm.auto import tqdm
import tarfile
import zipfile

# 2. Create temp file with list of processed filenames, with success or error condition
# 4. Create result file to log matching filenames
# 1. Load metadata stream from zip file
# 5. Check if temp file exists. If yes, get row of last processed filename
# 5. Start search metadata

data_dir = "/Volumes/Parler/"
output_dir = "../../data/"

parser = argparse.ArgumentParser(description="Function Variables")
parser.add_argument("inputFile", type=str)
parser.add_argument("searchKey", type=str)
args = parser.parse_args()

search_key = args.searchKey
input_file = args.inputFile

def check_output_files(output_file):
    # Create or Load Temp file
    with open(output_file) as o:
        lines = o.readlines()
    rows = len(lines)
    return (rows +1)

def catch_geolocation_data(jsonData, search_key):
    for key, value in jsonData.items():
        if key.startswith(search_key):
            return True
        else:
            continue
    # print('No location data found')
    return False

def process_metadata(filename, metadata_stream, output_file):
    contentObject = metadata_stream.extractfile(filename)
    
    with open(output_file, 'a+') as o:
        try:
            jsonContent = contentObject.read().decode('ascii')
            contentObject.close()
        except:
            try:
                jsonContent = contentObject.read()
                contentObject.close()
            except:
                o.write(f'{filename}, encoding_error\n')
                exit()

        try:
            data = json.loads(jsonContent)

            if catch_geolocation_data(data[0], search_key):
                o.write(f'{filename}, {search_key}_detected\n')
            else:
                o.write(f'{filename}, {search_key}_not_detected\n')
        except:
            o.write(f'{filename}, error\n')


def find_search_metadata_tar(input_file, search_key):
    metadata_stream = tarfile.open(input_file, 'r:gz')
    print(f"{input_file} loaded")
    
    fileNames = metadata_stream.getnames()
    jsonFiles = [fileName for fileName in fileNames if fileName.endswith('.json')]

    print(f'There are {len(jsonFiles)} json files in the tarfile.')

    output_file = os.path.join(output_dir, 'outputs', f'processed_files_{search_key}.txt')

    if os.path.exists(output_file):
        start_row = check_output_files(output_file)
    else:
        open(output_file, 'w').close()
        start_row = 0
    print(f"Starting from row {start_row}")

    pbar = tqdm(desc='Video Metadata', total=len(jsonFiles[start_row:]))
    
    for each in jsonFiles[start_row:]:
        process_metadata(each, metadata_stream, output_file)
        pbar.update(1)
 

if __name__ == '__main__':
    find_search_metadata_tar(args.inputFile, args.searchKey)
