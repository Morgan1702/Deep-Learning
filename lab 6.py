import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

# Завантаження CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Нормалізація
y_train, y_test = tf.keras.utils.to_categorical(y_train, 10), tf.keras.utils.to_categorical(y_test, 10)


# Функція для Inception-модуля
def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5,
                     filters_pool_proj):
    conv1x1 = layers.Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')(x)

    conv3x3 = layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
    conv3x3 = layers.Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')(conv3x3)

    conv5x5 = layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
    conv5x5 = layers.Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')(conv5x5)

    pool_proj = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = layers.Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu')(pool_proj)

    return layers.concatenate([conv1x1, conv3x3, conv5x5, pool_proj])


# Побудова GoogLeNet
input_layer = Input(shape=(32, 32, 3))
x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)
x = layers.Conv2D(192, (3, 3), padding='same', activation='relu')(x)
x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

# Додавання Inception модулів
x = inception_module(x, 64, 96, 128, 16, 32, 32)
x = inception_module(x, 128, 128, 192, 32, 96, 64)
x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

x = inception_module(x, 192, 96, 208, 16, 48, 64)
x = inception_module(x, 160, 112, 224, 24, 64, 64)
x = inception_module(x, 128, 128, 256, 24, 64, 64)
x = inception_module(x, 112, 144, 288, 32, 64, 64)
x = inception_module(x, 256, 160, 320, 32, 128, 128)
x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

x = inception_module(x, 256, 160, 320, 32, 128, 128)
x = inception_module(x, 384, 192, 384, 48, 128, 128)

x = layers.GlobalAveragePooling2D()(x)
output_layer = layers.Dense(10, activation='softmax')(x)

model = models.Model(input_layer, output_layer)
model.summary()

# Компіляція та тренування
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Оцінка та візуалізація
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Тестова точність: {test_acc * 100:.2f}%")

# Візуалізація кількох зображень
predictions = np.argmax(model.predict(x_test[:10]), axis=1)
true_labels = np.argmax(y_test[:10], axis=1)

plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i])
    plt.title(f"True: {true_labels[i]}\nPred: {predictions[i]}")
    plt.axis('off')
plt.show()
