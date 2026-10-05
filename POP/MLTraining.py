import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([1, 2, 3, 4, 5])
y = np.array([35, 40, 45, 50, 55])

X = x.reshape(-1, 1)

model = LinearRegression()
model.fit(X, y)

print("slope:", model.coef_[0])
print("Intercept:", model.intercept_)

predicted_y = model.predict(X)

print("Predicted:", predicted_y)
