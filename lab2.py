import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

# Завантаження та підготовка даних
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Перетворення зображень в 1D масив і нормалізація
train_images = train_images.reshape(train_images.shape[0], 28 * 28).astype('float32') / 255
test_images = test_images.reshape(test_images.shape[0], 28 * 28).astype('float32') / 255

# Приведення міток до one-hot форматування
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# Побудова моделі
mlp_model = Sequential([
    layers.Dense(512, activation='relu', input_shape=(28 * 28,)),
    layers.Dense(300, activation='relu'),  # змінено кількість нейронів
    layers.Dense(10, activation='softmax')  # вихідний шар з 10 класами
])

# Компіляція моделі
mlp_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Тренування моделі з валідаційним набором
training_history = mlp_model.fit(train_images, train_labels, epochs=10, batch_size=128, validation_split=0.15)  # Змінено валідацію

# Оцінка моделі на тестових даних
test_loss, test_accuracy = mlp_model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Прогнозування на тестових зображеннях
test_predictions = mlp_model.predict(test_images)

# Функція для відображення тестових зображень та їх передбачених міток
def display_images(images, actual_labels, predicted_labels, num=6):
    plt.figure(figsize=(12, 4))
    for i in range(num):
        plt.subplot(1, num, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        true_label = np.argmax(actual_labels[i])
        predicted_label = np.argmax(predicted_labels[i])
        plt.title(f"True: {true_label}\nPred: {predicted_label}")
        plt.axis('off')
    plt.show()

# Відображення кількох прикладів з тестового набору
display_images(test_images, test_labels, test_predictions)
