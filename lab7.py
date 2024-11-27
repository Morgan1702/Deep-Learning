import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
import matplotlib.pyplot as plt
import numpy as np

# Завантаження датасету CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Одна гаряча кодировка міток
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Визначення блоку ResNet
def resnet_block(inputs, filters, strides=1, downsample=False):
    x = layers.Conv2D(filters, 3, strides=strides, padding='same',
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, strides=1, padding='same',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    if downsample:
        inputs = layers.Conv2D(filters, 1, strides=strides, padding='same')(inputs)
    x = layers.Add()([x, inputs])
    return layers.ReLU()(x)

# Створення моделі ResNet-34
def build_resnet34(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, 7, strides=2, padding='same',
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Блоки ResNet
    for filters, reps, strides in [(64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)]:
        for i in range(reps):
            x = resnet_block(x, filters, strides if i == 0 else 1, downsample=(i == 0 and strides > 1))

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)

# Параметри моделі
model = build_resnet34(input_shape=(32, 32, 3), num_classes=10)
model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Тренування моделі
history = model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=128)

# Тестування моделі
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Точність на тестовому наборі: {test_acc:.2f}")

# Візуалізація прогнозів
predictions = np.argmax(model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Відображення кількох зображень із передбаченнями
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i])
    plt.title(f"True: {y_true[i]}, Pred: {predictions[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
