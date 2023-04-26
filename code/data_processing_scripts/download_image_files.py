import pandas as pd
import time
from tqdm.auto import tqdm
import wget

data = '/Volumes/Parler/data/ddosecrets-parler-images-listing.csv'
output_dir = '/Volumes/Parler/downloads/images/'

base_url = 'https://s3.wasabisys.com/ddosecrets-parler-images/'

# Get list of images

df = pd.read_csv(data, header=0, names=['date', 'time', 'size', 'filename'], on_bad_lines='skip')
files = df['filename'].tolist()

# Download files to external harddrive

last_downloaded = len(os.listdir(output_dir))-1
print(f'Starting at file {last_downloaded}')

pbar = tqdm(desc='Download Image Files', total=len(files[last_downloaded:]))

with open('../../data/outputs/image_files.csv', 'a+') as o:
    for each in files[last_downloaded:]:
        url = base_url + each
        try:
            wget.download(url, output_dir)
            o.write(f"{each}, success\n")
        except ConnectionResetError:
            print("==> ConnectionResetError")
            o.write(f"{each}, connection_error\n")
            pass
        except: 
            print("==> Error")
            o.write(f"{each}, error\n")
            pass
        pbar.update(1)
        time.sleep(1)