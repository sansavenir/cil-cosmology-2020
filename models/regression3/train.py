import os
import numpy as np
import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics
import features
import pandas as pd
import getpass
import argparse

USERNAME = getpass.getuser()
SCRATCH_DIR = os.path.join('/cluster/scratch', USERNAME)
DATA_DIR = os.path.join(SCRATCH_DIR, 'cosmology_aux_data_170429/cosmology_aux_data_170429/')

parser = argparse.ArgumentParser(description='reg')
parser.add_argument('--data_dir', type=str, default=DATA_DIR,
                    help='Where the cosmology dataset resides')
parser.add_argument('--val', type=int, default=1,
                    help='Flag indicating whether to validate')
parser.add_argument('--pred', type=int, default=1,
                    help='Flag indicating whether to prepare a submission')
parser.add_argument('--cached_features', type=int, default=0,
                    help='Flag indicating whether to use cached features')
args = parser.parse_args()


def load_query(root_dir):
    query_path = os.path.join(root_dir, 'query')
    image_names = os.listdir(query_path)
    image_names = np.asarray([os.path.splitext(n)[0] for n in image_names])

    paths = [os.path.join(query_path, n + '.png') for n in image_names]

    return paths, image_names


def load_scored(root_dir):
    scored_path = os.path.join(root_dir, 'scored.csv')
    data = np.genfromtxt(scored_path, delimiter=',', skip_header=1, dtype=np.float32)

    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    data = data[idx]

    names = data[:, 0]
    scores = data[:, 1]
    paths = [os.path.join(root_dir, 'scored', str(int(n)) + '.png') for n in names]

    return paths, scores


def save_submission(output_path, image_names, scores):
    res = np.vstack([image_names, scores]).T
    df = pd.DataFrame(res, columns=['Id', 'Predicted'])
    df.to_csv(output_path, index=False)


def train(fs, gt, split=0.9):
    num_samples = fs.shape[0]
    num_train_samples = int(split * num_samples)

    fs_train = fs[:num_train_samples]
    fs_test = fs[num_train_samples:]
    gt_train = gt[:num_train_samples]
    gt_test = gt[num_train_samples:]

    model = sklearn.ensemble.RandomForestRegressor(criterion='mae',
                                                   oob_score=True,
                                                   n_estimators=100,
                                                   verbose=5,
                                                   n_jobs=-1)
    # model = sklearn.model_selection.GridSearchCV(model,
    #                                              {'n_estimators': [80, 85, 90, 95, 100, 105, 110]},
    #                                              verbose=5,
    #                                              scoring='neg_mean_absolute_error',
    #                                              n_jobs=-1)

    model.fit(fs_train, gt_train)

    if fs_test.shape[0] > 0:
        pred = model.predict(fs_test)
        mae = sklearn.metrics.mean_absolute_error(gt_test, pred)
        print('VALIDATION MAE:', mae)

        l = range(0, 8)
        r = range(1, 9)
        for l, r in zip(l, r):
            mask = (gt_test >= l) & (gt_test < r)
            count = np.count_nonzero(mask)
            if count == 0:
                print('No samples in', l, '-', r)
            else:
                mae = sklearn.metrics.mean_absolute_error(gt_test[mask], pred[mask])
                print('MAE of', count, 'samples in', l, '-', r, ': ', mae)


    return model


def main():
    paths, scores = load_scored(args.data_dir)

    fs_path = os.path.join(args.data_dir, 'features.npy')
    ss_path = os.path.join(args.data_dir, 'scores.npy')
    if os.path.exists(fs_path) and args.cached_features:
        print("Loading features from", fs_path)
        fs = np.load(fs_path)
        ss = np.load(ss_path)
    else:
        fs, ss = features.get_train_features(paths, scores)
        np.save(fs_path, fs)
        np.save(ss_path, ss)

    split = 0.8 if args.val else 1
    model = train(fs, ss, split=split)

    if args.pred:
        paths, image_names = load_query(args.data_dir)
        fs = features.get_pred_features(paths)
        ss = model.predict(fs)
        output_path = os.path.join(args.data_dir, 'pred.csv')
        save_submission(output_path, image_names, ss)


main()