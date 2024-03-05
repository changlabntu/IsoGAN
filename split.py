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
if 0:
    df = pd.read_csv('env/csv/womac4_moaks.csv')
    df = df.loc[df['V$$WOMKP#'] > 0, :]
    df.reset_index(inplace=True)
    train_index = list(df.loc[df['READPRJ'].isnull(), :].index)[:]
    test_index = list(df.loc[df['READPRJ'].notnull(), :].index)[:]

# new indicies
x = pd.read_csv('env/csv/womac4_moaks.csv')
labels = (x.loc[x['SIDE'] == 'RIGHT']['V$$WOMKP#']).values > (x.loc[x['SIDE'] == 'LEFT']['V$$WOMKP#']).values
labels = [int(x) for x in labels]
# labels = [(int(x),) for x in labels]
knee_painful = x.loc[(x['V$$WOMKP#'] > 0)].reset_index()
pmindex = knee_painful.loc[~knee_painful['READPRJ'].isna()].index.values
ID_has_eff = x.loc[~x['V$$MEFFWK'].isna()]['ID'].unique()
pmeffid = knee_painful.loc[knee_painful['ID'].isin(ID_has_eff)].index.values

train_index = knee_painful.loc[~knee_painful['ID'].isin(ID_has_eff)].index.values
test_index = pmeffid


# path
source = os.environ.get('DATASET') + 'womac4/'
a_full = sorted(glob.glob(source + 'full/ap/*'))
b_full = sorted(glob.glob(source + 'full/bp/*'))

# remove directories on womac4/train and womac4/val if they exist
for folder in ['train', 'val']:
    for subfolder in ['ap', 'bp']:
        try:
            shutil.rmtree(source + folder + '/' + subfolder)
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
print('Length of train and val subjects:', len(train_index), len(test_index))
print('Length of train and val slices:', 23 * len(train_index), 23 * len(test_index))
print('Length of a train and val:', len(a_train), len(a_val))
print('Length of b train and val:', len(b_train), len(b_val))
print('asserted non-overlapping slices done')

# assert no overlapping in subjects
a_train_subjects = set([x.split('/')[-1].split('_')[0] for x in a_train])
b_train_subjects = set([x.split('/')[-1].split('_')[0] for x in b_train])
a_val_subjects = set([x.split('/')[-1].split('_')[0] for x in a_val])
b_val_subjects = set([x.split('/')[-1].split('_')[0] for x in b_val])

assert len(a_train_subjects.intersection(a_val_subjects)) == 0
assert len(b_train_subjects.intersection(b_val_subjects)) == 0
print('asserted non-overlapping subjects done')