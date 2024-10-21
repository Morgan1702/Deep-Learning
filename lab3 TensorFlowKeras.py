import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

# Завантаження даних MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Перетворення та нормалізація
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255

# One-hot encoding мітки
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# Створення моделі LeNet-5
lenet_model = Sequential([
    layers.Conv2D(6, (5, 5), activation='tanh', input_shape=(28, 28, 1), padding='same'),
    layers.AveragePooling2D(),
    layers.Conv2D(16, (5, 5), activation='tanh'),
    layers.AveragePooling2D(),
    layers.Conv2D(120, (5, 5), activation='tanh'),
    layers.Flatten(),
    layers.Dense(84, activation='tanh'),
    layers.Dense(10, activation='softmax')
])

# Компіляція моделі
lenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Тренування моделі
lenet_model.fit(train_images, train_labels, epochs=10, batch_size=128, validation_split=0.1)

# Оцінка на тестовому наборі
test_loss, test_accuracy = lenet_model.evaluate(test_images, test_labels)
print(f'Test accuracy (TensorFlow): {test_accuracy:.4f}')

# Прогноз на тестовому наборі
test_predictions = lenet_model.predict(test_images)

# Функція для відображення зображень
def plot_mnist_images(images, labels, preds, num=5):
    plt.figure(figsize=(10, 4))
    for i in range(num):
        plt.subplot(1, num, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        true_label = np.argmax(labels[i])
        pred_label = np.argmax(preds[i])
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis('off')
    plt.show()

# Відображення кількох прикладів з тестового набору
plot_mnist_images(test_images, test_labels, test_predictions)
