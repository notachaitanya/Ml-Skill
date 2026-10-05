from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

for name, model in [('Linear', LinearRegression()), ('Ridge', Ridge(alpha=1.0)), ('Lasso', Lasso(alpha=0.1))]:
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    print(name, "R2:", r2_score(y_test, pred), "MSE:", mean_squared_error(y_test, pred))