import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Завантаження даних
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Розбивка даних на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Створення моделі лінійної регресії та навчання
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# Прогнозування на тестовій вибірці
y_pred = regr.predict(X_test)

# Розрахунок коефіцієнтів регресії та показників
print("Regression coefficients:", regr.coef_)
print("Intercept:", regr.intercept_)
print("Mean absolute error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean squared error (MSE):", mean_squared_error(y_test, y_pred))
print("R2 score:", r2_score(y_test, y_pred))

# Побудова графіка
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
plt.show()
