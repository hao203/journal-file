import pathlib
import random
import tensorflow as tf

data_root = pathlib.Path(r'C:\Users\Eliot Drizzle\Documents\data')
all_images_paths = list(data_root.glob('*/*'))

all_images_paths = [str(path) for path in all_images_paths]
random.shuffle(all_images_paths)

label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
label_to_index = dict((name,index) for index,name in enumerate(label_names))
all_images_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_images_paths]

def preprocess_img(image):
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize(image,(28,28))
    image /= 255.0
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_img(image)

AUTOTUNE = tf.data.experimental.AUTOTUNE
Batch_size = 15

size = int(len(all_images_paths) * 0.1)

path_ds_train = tf.data.Dataset.from_tensor_slices(all_images_paths[:-size])
image_ds_train = path_ds_train.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
label_ds_train = tf.data.Dataset.from_tensor_slices(tf.cast(all_images_labels[:-size],tf.int64))
ds_train = tf.data.Dataset.zip((image_ds_train,label_ds_train))

ds_train = ds_train.batch(Batch_size)
ds_train = ds_train.prefetch(buffer_size=AUTOTUNE)

path_ds_test = tf.data.Dataset.from_tensor_slices(all_images_paths[-size:])
image_ds_test = path_ds_test.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
label_ds_test = tf.data.Dataset.from_tensor_slices(tf.cast(all_images_labels[-size:],tf.int64))
ds_test = tf.data.Dataset.zip((image_ds_test,label_ds_test))

ds_test = ds_test.batch(Batch_size)
ds_test = ds_test.prefetch(buffer_size=AUTOTUNE)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same', strides=(2, 2)))

model.add(tf.keras.layers.Conv2D(128, (5, 5), padding='same', strides=(1, 1), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same', strides=(2, 2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])

model.fit(ds_train, epochs=10)
loss, acc = model.evaluate(ds_test)
