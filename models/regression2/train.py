import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import applications as apps
import functools as ft
import glob
import pandas as pd
from classification_models.tfkeras import Classifiers

ResNet18, preprocess_input = Classifiers.get('resnet18')

np.random.seed(123)

SCRATCH_DIR = '/cluster/scratch/laurinb/'
DATA_DIR = '/cluster/scratch/laurinb/cil-cosmology-2020/data'
NUM_EPOCHS = 100
BATCH_SIZE = 64

INTERVALS = np.asarray([
    [0, 2],
    [2, 4],
    [4, 8]
])

def err_distr(y_true, y_pred, idx):
    l = range(0, 8)
    r = range(1, 9)
    
    l, r = list(zip(l, r))[idx]
    mask = (tf.greater(y_true, l) & tf.less_equal(y_true, r))
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    mae = tf.where(tf.math.is_nan(mae), tf.zeros_like(mae), mae)
    
    return mae


def lmin(y_true, y_pred):
    return tf.reduce_min(y_true)


def lmax(y_true, y_pred):
    return tf.reduce_max(y_true)


def smae(y_true, y_pred):
    loss = tf.keras.losses.mean_absolute_error(y_true, y_pred)

    mult = tf.where(y_true < 1, tf.constant(1, dtype=y_true.dtype), y_true)
    loss *= mult

    return loss


def create_backbone(num_classes):
    # weights = '/cluster/scratch/laurinb/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # model = apps.ResNet50(classes=num_classes,
    #                       weights=weights,
    #                       include_top=False,
    #                       input_shape=(250, 250, 3),
    #                       pooling='avg')

    weights = '/cluster/scratch/laurinb/resnet18_imagenet_1000_no_top.h5'
    model = ResNet18((125, 125, 3),
                     weights=weights,
                     include_top=False,
                     classes=num_classes)
    avg = tf.keras.layers.GlobalAveragePooling2D()(model.output)
    model = tf.keras.models.Model(inputs=model.inputs, outputs=avg)

    return model


def create_regression(num_classes, ckpt=None):
    model = create_backbone(num_classes)
    print('REGRESSION WITH', num_classes, 'CLASSES AND BACKBONE', model)

    regression = tf.keras.layers.Dense(1, activation='linear')(model.outputs[0])
    model = tf.keras.models.Model(inputs=model.inputs, outputs=regression)

    err_distr_metrics = [ft.partial(err_distr, idx=i) for i in range(8)]
    for idx, func in enumerate(err_distr_metrics):
        func.__name__ = str(idx) 
    
    if ckpt:
        model.load_weights(ckpt)
    model.compile(optimizer='adam',
                  loss=['mae'],
                  metrics=['mae']+err_distr_metrics)

    return model


def create_classifier(num_classes, ckpt=None):
    print('CLASSIFICATION WITH', num_classes, 'CLASSES FOR INTERVALS:', INTERVALS)

    model = create_backbone(num_classes)
    cls_id = tf.keras.layers.Dense(num_classes, activation='softmax')(model.outputs[0])
    model = tf.keras.models.Model(inputs=model.inputs, outputs=cls_id)
    
    if ckpt:
        model.load_weights(ckpt)
    model.compile(optimizer='adam',
                  loss=['categorical_crossentropy'],
                  metrics=['acc'])

    return model


def create_dataset(root, train=True):
    # labeled_path = os.path.join(root, 'labeled.csv')
    # labeled = np.genfromtxt(labeled_path, delimiter=',', skip_header=1, dtype=np.float32)
    # labeled = labeled[np.where(labeled[:, 1] > 0)]
    # labeled = np.hstack([labeled, 8 * np.ones([labeled.shape[0], 1])])

    scored_path = os.path.join(root, 'scored.csv')
    scored = np.genfromtxt(scored_path, delimiter=',', skip_header=1, dtype=np.float32)
    scored[:, 1] = scored[:, 1]
    scored = np.hstack([scored, np.zeros([scored.shape[0], 1])])

    # data = np.vstack([labeled, scored])
    # mask = scored[:, 1] < 2
    # data = scored[mask]
    data = scored

    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    data = data[idx]

    num_train_samples = int(0.9 * data.shape[0])
    if train:
        data = data[:num_train_samples]
    else:
        data = data[num_train_samples:]

    names = data[:, 0]
    scores = data[:, 1]
    is_labeled = data[:, 2]
    paths = [os.path.join(root, 'labeled' if l else 'scored', str(int(n)) + '.png') for (n, l) in zip(names, is_labeled)]

    return tf.data.Dataset.from_tensor_slices((paths, scores)), data.shape[0]


def img_preprocess(x):
    x = tf.compat.v1.read_file(x)
    x = tf.io.decode_image(x, channels=1)

    x = tf.image.grayscale_to_rgb(x)
    x = tf.cast(x, tf.float32) * (1. / 255.0)
    x = tf.reshape(x, [1000, 1000, 3])
    x = preprocess_input(x)

    return x


def query_preprocess(x):
    x = img_preprocess(x)

    return x

    # x0 = x[:500, :500]
    # x1 = x[:500, 500:]
    # x2 = x[500:, :500]
    # x3 = x[500:, 500:]

    # return tf.stack([x0, x1, x2, x3])


def val_preprocess(x, y):
    x = img_preprocess(x)
    x = tf.image.random_crop(x, [125, 125, 3])

    return x, y


def train_preprocess(x, y):
    x, y = val_preprocess(x, y)

    x = tf.image.random_flip_left_right(x)

    return x, y


def train():
    # model = create_classifier(2)
    model = create_regression(50)

    train_dataset, num_train_samples = create_dataset(DATA_DIR, train=True)
    train_dataset = train_dataset.shuffle(num_train_samples).map(train_preprocess).batch(BATCH_SIZE)

    test_dataset, _ = create_dataset(DATA_DIR, train=False)
    test_dataset = test_dataset.map(val_preprocess).batch(BATCH_SIZE)

    now = datetime.datetime.now()
    date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
    checkpoint_path = os.path.join(SCRATCH_DIR, 'regression', date_time, 'ckpt.ckpt')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     monitor='val_mae',
                                                     save_best_only=True,
                                                     save_weights_only=True,
                                                     verbose=1)

    model.fit(train_dataset,
              epochs=NUM_EPOCHS,
              validation_data=test_dataset,
              validation_freq=1,
              shuffle=True,
              callbacks=[cp_callback])
    model.evaluate(test_dataset)


def pred():
    root_dir = '/cluster/scratch/laurinb/regression/07-23-2020-10-36-29'
    ckpt_path = os.path.join(root_dir, 'ckpt.ckpt')
    model = create_regression(50, ckpt=ckpt_path)

    query_path = os.path.join(DATA_DIR, 'query')
    image_names = os.listdir(query_path)
    image_names = np.asarray([os.path.splitext(n)[0] for n in image_names])

    paths = [os.path.join(query_path, n + '.png') for n in image_names]
    dataset = tf.data.Dataset.from_tensor_slices(paths).map(query_preprocess).batch(1)

    scores = model.predict(dataset, verbose=1)
    scores = np.mean(scores, axis=-1)

    scores = np.clip(scores, 0, 8)
    scores = np.squeeze(scores)

    res = np.vstack([image_names, scores]).T
    print(scores.shape, res.shape)
    output_path = os.path.join(root_dir, 'pred.csv')
    df = pd.DataFrame(res, columns=['Id', 'Predicted'])
    df.to_csv(output_path, index=False)


train()



