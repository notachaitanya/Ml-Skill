import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = sns.load_dataset('mpg').dropna()

df = pd.get_dummies(df, columns=['origin'], drop_first=True)
X = df.drop(columns=['mpg', 'name'])
y = df['mpg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_s, y_train)
pred = model.predict(X_test_s)

print("R2:", r2_score(y_test, pred))
print("MSE:", mean_squared_error(y_test, pred))