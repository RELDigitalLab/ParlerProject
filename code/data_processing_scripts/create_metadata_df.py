import argparse
import json
import os
import pandas as pd
from tqdm.auto import tqdm
import tarfile

# 1. Load file report
# 2. List of matching files
# 3. For each in list, load metadata json
# 4. Create dataframe of matching files

source_file = "/Volumes/Parler/data/metadata.tar.gz"
data_dir = "../../data/"

parser = argparse.ArgumentParser(description="Function Variables")
parser.add_argument("inputFile", type=str)
parser.add_argument("responseKey", type=str)
args = parser.parse_args()

search_key = args.responseKey
input_file = args.inputFile



