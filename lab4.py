import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

# Завантаження та нормалізація даних CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Визначення архітектури мережі AlexNet з урахуванням розміру CIFAR-10
def create_alexnet():
    model = models.Sequential([
        layers.Conv2D(96, (3, 3), strides=(1, 1), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        
        layers.Conv2D(384, (3, 3), activation='relu'),
        layers.Conv2D(384, (3, 3), activation='relu'),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Ініціалізація моделі AlexNet
model = create_alexnet()

# Компіляція моделі
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Тренування моделі
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Оцінка якості моделі на тестовому наборі
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Точність на тестовому наборі: {test_acc * 100:.2f}%")

# Візуалізація результатів для кількох тестових зображень
class_names = ['Літак', 'Автомобіль', 'Птах', 'Кіт', 'Олень', 'Собака', 'Жаба', 'Кінь', 'Корабель', 'Вантажівка']

# Відображення кількох зображень та їх передбачуваних і реальних міток
num_images = 5
plt.figure(figsize=(10, 5))
for i in range(num_images):
    plt.subplot(1, num_images, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    img = x_test[i]
    plt.imshow(img, cmap=plt.cm.binary)
    true_label = class_names[y_test[i][0]]
    
    # Корекція прогнозу зображення
    predicted_label = class_names[np.argmax(model.predict(np.expand_dims(img, axis=0)), axis=1)[0]]
    
    plt.xlabel(f"Істинно: {true_label}\nПрогноз: {predicted_label}")
plt.show()
