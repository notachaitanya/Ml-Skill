from sklearn.datasets import fetch_openml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = fetch_openml(name='adult', version=2, as_frame=True)
df = data.frame

print(df.shape)
df.isnull().sum()

# EDA
sns.countplot(x='class', data=df); plt.title('Income Class Distribution'); plt.show()
sns.histplot(df['age'], bins=30); plt.title('Age Distribution'); plt.show()
sns.countplot(y='education', data=df, order=df['education'].value_counts().index); plt.show()

# Feature engineering
df['capital_diff'] = df['capital-gain'] - df['capital-loss']
df_encoded = pd.get_dummies(df, columns=['workclass','education','marital-status','occupation','relationship','race','sex','native-country'], drop_first=True)
print(df_encoded.shape)