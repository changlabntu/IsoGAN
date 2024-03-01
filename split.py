import pandas as pd
import argparse
from dotenv import load_dotenv
import os, glob, shutil
from tqdm import tqdm

parser = argparse.ArgumentParser()  # add_help=False)
parser.add_argument('--env', type=str, default='t09b', help='environment_to_use')
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=str, default='dummy')
args = parser.parse_args()

if args.env is not None:
    load_dotenv('env/.' + args.env)
else:
    load_dotenv('env/.t09')

# indices
df = pd.read_csv('env/csv/womac4_moaks.csv')
df = df.loc[df['V$$WOMKP#'] > 0, :]
df.reset_index(inplace=True)
train_index = list(df.loc[df['READPRJ'].isnull(), :].index)[:]
test_index = list(df.loc[df['READPRJ'].notnull(), :].index)[:]

# path
source = os.environ.get('DATASET') + 'womac4/'
a_full = sorted(glob.glob(source + 'full/ap/*'))
b_full = sorted(glob.glob(source + 'full/bp/*'))

# remove directories on womac4/train and womac4/val if they exist
for folder in ['train', 'val']:
    for subfolder in ['ap', 'bp']:
        try:
            os.rmdir(source + folder + '/' + subfolder)
        except:
            pass
# make directories on womac4/train and womac4/val
for folder in ['train', 'val']:
    for subfolder in ['ap', 'bp']:
        os.makedirs(source + folder + '/' + subfolder, exist_ok=True)

# copy files from womac4/full/ to womac4/train/ and womac4/val/
for i in tqdm(train_index):
    for z in range(23*i, 23*(i+1)):
        shutil.copy(a_full[z], source + 'train/ap/' + a_full[z].split('/')[-1])
        shutil.copy(b_full[z], source + 'train/bp/' + b_full[z].split('/')[-1])

for i in tqdm(test_index):
    for z in range(23*i, 23*(i+1)):
        shutil.copy(a_full[z], source + 'val/ap/' + a_full[z].split('/')[-1])
        shutil.copy(b_full[z], source + 'val/bp/' + b_full[z].split('/')[-1])

# make sure there is no overlapping files between train and val
a_train = sorted(glob.glob(source + 'train/ap/*'))
b_train = sorted(glob.glob(source + 'train/bp/*'))
a_val = sorted(glob.glob(source + 'val/ap/*'))
b_val = sorted(glob.glob(source + 'val/bp/*'))
assert len(set(a_train).intersection(set(a_val))) == 0
assert len(set(b_train).intersection(set(b_val))) == 0
print('assert done')
