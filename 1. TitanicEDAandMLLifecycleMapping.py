import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = sns.load_dataset('titanic')
print(df.shape)
df.info()
df.isnull().sum()

# EDA
sns.countplot(x='survived', data=df); plt.title('Survival Count'); plt.show()
sns.countplot(x='class', hue='survived', data=df); plt.title('Survival by Class'); plt.show()
sns.countplot(x='sex', hue='survived', data=df); plt.title('Survival by Sex'); plt.show()
sns.histplot(df['age'].dropna(), bins=30); plt.title('Age Distribution'); plt.show()

# ML Lifecycle mapping (comment, no code needed)
# Problem Definition -> Data Collection -> EDA -> Preprocessing ->
# Feature Engineering -> Model Selection -> Training -> Evaluation -> Deployment