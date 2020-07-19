import os
import datetime
import numpy as np
import tensorflow as tf
# from tensorflow.python.keras import applications as apps
from tensorflow.keras import applications as apps

np.random.seed(123)

DATA_DIR = '/kaggle/input/cosmology/cosmology_aux_data_170429/cosmology_aux_data_170429'
NUM_EPOCHS = 30
BATCH_SIZE = 8
BASE_LEARNING_RATE = 0.1
LR_SCHEDULE = [(0.1, 10), (0.01, 20)]
L2_WEIGHT_DECAY = 2e-4


def create_model(num_classes=20, ckpt=None):
    init_weights = '/kaggle/input/checkpoints/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    
    model = apps.ResNet50(classes=num_classes,
                          weights=init_weights,
                          include_top=False,
                          input_shape=(1000, 1000, 3),
                          pooling='avg')

    regression = tf.keras.layers.Dense(1, activation='linear')(model.outputs[0])
    model = tf.keras.models.Model(inputs=model.inputs, outputs=regression)

    if ckpt:
        model.load_weights(ckpt)
    model.compile(optimizer='adam',
                  loss=['huber'],
                  metrics=['mean_absolute_error'])

    return model


def create_dataset(root, train=True):
    labeled_path = os.path.join(root, 'labeled.csv')
    labeled = np.genfromtxt(labeled_path, delimiter=',', skip_header=1, dtype=np.float32)
    labeled = labeled[np.where(labeled[:, 1] > 0)]
    labeled = np.hstack([labeled, 8*np.ones([labeled.shape[0], 1])])

    scored_path = os.path.join(root, 'scored.csv')
    scored = np.genfromtxt(scored_path, delimiter=',', skip_header=1, dtype=np.float32)
    scored[:, 1] = scored[:, 1]
    scored = np.hstack([scored, np.zeros([scored.shape[0], 1])])

    data = np.vstack([labeled, scored])
        
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


def preprocess(x, y, train=True):
    x = tf.compat.v1.read_file(x)
    x = tf.image.decode_jpeg(x, dct_method="INTEGER_ACCURATE")
    x = tf.cast(x , tf.float32) * (1. / 255.0)
#     x = tf.image.resize(x, [255, 255])
    x = tf.image.grayscale_to_rgb(x)

    if train:
        x = tf.image.random_flip_left_right(x)
 
    return x, y


def schedule(epoch):
    initial_learning_rate = BASE_LEARNING_RATE
    learning_rate = initial_learning_rate
    for mult, start_epoch in LR_SCHEDULE:
        if epoch >= start_epoch:    
            learning_rate = initial_learning_rate * mult
        else:
            break
            
    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
  
    return learning_rate


def main():
    model = create_model()

    train_dataset, num_train_samples = create_dataset(DATA_DIR, train=True)
    train_dataset = train_dataset.shuffle(num_train_samples).map(preprocess).batch(BATCH_SIZE, drop_remainder=True)

    test_dataset, _ = create_dataset(DATA_DIR, train=False)
    test_dataset = test_dataset.map(preprocess).batch(BATCH_SIZE, drop_remainder=True)

    lr_schedule_callback = tf.keras.callbacks.LearningRateScheduler(schedule)
    
    log_dir="/kaggle/working/cosmology/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
    file_writer.set_as_default()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir,
      update_freq='batch',
      histogram_freq=1)

    model.fit(train_dataset,
              epochs=NUM_EPOCHS,
              validation_data=test_dataset,
              validation_freq=1,
              callbacks=[lr_schedule_callback, tensorboard_callback])
    model.evaluate(test_dataset)


main()

