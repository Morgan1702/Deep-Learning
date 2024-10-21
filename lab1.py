import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Генерація даних
rng = np.random.default_rng(seed=1)  
x_vals = np.linspace(0, 6, 200)  
y_vals = np.sin(x_vals) + np.sin(6 * x_vals) + rng.normal(0, 0.1, x_vals.shape[0]) 

# Розділення на тренувальні та тестові дані
X_train, X_test, Y_train, Y_test = train_test_split(x_vals.reshape(-1, 1), y_vals, test_size=0.3, random_state=42)

# Налаштування параметрів для моделі MLPRegressor
param_grid = {
    'hidden_layer_sizes': [(40,), (80,), (40, 40), (80, 80)],  
    'activation': ['relu', 'logistic'],  
    'solver': ['adam', 'lbfgs'],  
    'learning_rate_init': [0.001, 0.005, 0.01],  
    'max_iter': [600, 1200] 
}

# Ініціалізація та пошук найкращих параметрів
model = MLPRegressor(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, Y_train)

# Отримання оптимальних параметрів
best_hyperparams = grid_search.best_params_
print(f'Оптимальні гіперпараметри: {best_hyperparams}')

# Навчання моделі з найкращими параметрами та більшою кількістю ітерацій
optimized_model = MLPRegressor(
    hidden_layer_sizes=best_hyperparams['hidden_layer_sizes'],
    activation=best_hyperparams['activation'],
    solver=best_hyperparams['solver'],
    learning_rate_init=best_hyperparams['learning_rate_init'],
    max_iter=2000,  
    random_state=42
)

optimized_model.fit(X_train, Y_train)

# Прогноз для тренувальних та тестових даних
train_pred = optimized_model.predict(X_train)
test_pred = optimized_model.predict(X_test)

# Обчислення метрик для обох наборів даних
train_mse = mean_squared_error(Y_train, train_pred)
test_mse = mean_squared_error(Y_test, test_pred)
train_r2 = r2_score(Y_train, train_pred)
test_r2 = r2_score(Y_test, test_pred)

print(f'Помилка на тренуванні (MSE): {train_mse}, R2: {train_r2}')
print(f'Помилка на тесті (MSE): {test_mse}, R2: {test_r2}')

# Візуалізація результатів
plt.figure(figsize=(12, 6))

# Тренувальні дані та їх прогнози
plt.subplot(1, 2, 1)
plt.scatter(X_train, Y_train, color='purple', label='Train Data')
plt.plot(X_train, train_pred, color='orange', label='Prediction')
plt.title('Тренувальні дані vs Прогноз')
plt.legend()

# Тестові дані та їх прогнози
plt.subplot(1, 2, 2)
plt.scatter(X_test, Y_test, color='blue', label='Test Data')
plt.plot(X_test, test_pred, color='red', label='Prediction')
plt.title('Тестові дані vs Прогноз')
plt.legend()

plt.tight_layout()
plt.show()