import seaborn as sns
import pandas as pd

df = sns.load_dataset('titanic')

# Drop columns with too many missing values / redundant info
df_clean = df.drop(columns=['deck', 'embark_town', 'alive', 'who', 'adult_male', 'class'])

# Fill missing values
df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
df_clean['embarked'] = df_clean['embarked'].fillna(df_clean['embarked'].mode()[0])

# Encode categorical columns
df_clean = pd.get_dummies(df_clean, columns=['sex', 'embarked'], drop_first=True)

print(df_clean.isnull().sum())
print(df_clean.shape)
df_clean.to_csv('titanic_cleaned.csv', index=False)