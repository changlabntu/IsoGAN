import pandas as pd
import os, glob
from functools import reduce


def customize_data_split(args):
    dataset = args.dataset
    split = args.split
    if split is not None:
        folder = '/full/'

        first_paired_folders = [os.path.join(os.environ.get('DATASET') + args.dataset + folder, x) for x in args.direction.split('%')[0].split('_')]
        image_list = [sorted([x.split('/')[-1] for x in glob.glob(first_paired_folders[i] + '/*')]) for i in range(len(first_paired_folders))]
        length_data = len(reduce(set.intersection, [set(item) for item in image_list]))

        if split in ['x', 'y', 'small', 'all']:  # this works for load 2d slices individually
            if split == 'x':
                train_index = range(0, length_data // 10 * 7)
                test_index = range(length_data // 10 * 7, length_data)
            if split == 'y':
                train_index = range(length_data // 10 * 3, length_data)
                test_index = range(0, length_data // 10 * 3)
            if split == 'small':
                train_index = range(0, length_data // 10)
                test_index = range(length_data // 10, length_data)
            if split == 'all':
                train_index = range(0, length_data)
                test_index = range(0, 1)
        else:  # this works for load 3d slices
            if dataset == 'womac3':
                if split == 'moaks':
                    df = pd.read_csv('env/csv/womac3.csv')
                    train_index = [x for x in range(df.shape[0]) if not df['has_moaks'][x]]
                    test_index = [x for x in range(df.shape[0]) if df['has_moaks'][x]]
                elif split == 'a':
                    train_index = range(213, 710)
                    test_index = range(0, 213)
                elif split == 'b':
                    train_index = range(0, 497)
                    test_index = range(497, 710)
            if dataset == 'womac4':
                if split == 'a':
                    train_index = range(667, 2225)
                    test_index = range(333, 667)
                if split == 'moaks':
                    df = pd.read_csv('env/csv/womac4_moaks.csv')
                    df = df.loc[df['V$$WOMKP#'] > 0, :]
                    df.reset_index(inplace=True)
                    train_index = list(df.loc[df['READPRJ'].isnull(), :].index)[:]
                    test_index = list(df.loc[df['READPRJ'].notnull(), :].index)[:]
            if dataset == 'oaiseg':
                if split == 'a':
                    if args.load3d:
                        train_index = range(0, 70)
                        test_index = range(70, 88)
                    else:
                        train_index = range(2155, 9924) # 7769
                        test_index = range(0, 2155)

    else:
        folder = '/train/'
        train_index = None
        test_index = None
    return folder, train_index, test_index