import os
import numpy as np
import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics
import features
import pandas as pd
import getpass

USERNAME = getpass.getuser()
SCRATCH_DIR = os.path.join('/cluster/scratch', USERNAME)
DATA_DIR = os.path.join(SCRATCH_DIR, 'cosmology_aux_data_170429/cosmology_aux_data_170429/')


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
                                                   oob_score=True)
    model = sklearn.model_selection.GridSearchCV(model,
                                                 {'n_estimators': [5, 10, 50, 100]},
                                                 verbose=5,
                                                 scoring='neg_mean_absolute_error',
                                                 n_jobs=-1)
    model.fit(fs_train, gt_train)

    pred = model.predict(fs_test)
    mae = sklearn.metrics.mean_absolute_error(gt_test, pred)
    print(mae)

    return model


def main():
    paths, scores = load_scored(DATA_DIR)

    fs_path = os.path.join(SCRATCH_DIR, 'features.npy')

    if os.path.exists(fs_path):
        fs = np.load(fs_path)
    else:
        fs = features.get_features(paths)
        np.save(fs_path, fs)

    model = train(fs, scores)

    paths, image_names = load_query(DATA_DIR)
    fs = features.get_features(paths)
    scores = model.predict(fs)
    output_path = os.path.join(SCRATCH_DIR, 'pred.csv')
    save_submission(output_path, image_names, scores)


main()