import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Генерація випадкових даних
m = 100
X = np.linspace(-3, 3, m).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.uniform(-0.5, 0.5, m)

# Побудова лінійної регресії
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

# Побудова поліноміальної регресії
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

# Виведення коефіцієнтів
print("Linear regression coefficients:", lin_reg.coef_)
print("Linear regression intercept:", lin_reg.intercept_)
print("Polynomial regression coefficients:", poly_reg.coef_)
print("Polynomial regression intercept:", poly_reg.intercept_)

# Побудова графіків
plt.scatter(X, y, color='blue', label='Дані')
plt.plot(X, y_pred_lin, color='red', label='Лінійна регресія')
plt.plot(X, y_pred_poly, color='green', label='Поліноміальна регресія')
plt.xlabel('$x_1$')
plt.ylabel('$y$')
plt.legend()
plt.show()

