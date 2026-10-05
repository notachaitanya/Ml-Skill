from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import shap

data = fetch_openml(name='heart-statlog', version=1, as_frame=True)
df = data.frame
X = df.drop(columns=['class'])
y = LabelEncoder().fit_transform(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb = XGBClassifier(eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
print("XGBoost Accuracy:", accuracy_score(y_test, xgb.predict(X_test)))

lgbm = LGBMClassifier(random_state=42)
lgbm.fit(X_train, y_train)
print("LightGBM Accuracy:", accuracy_score(y_test, lgbm.predict(X_test)))

# SHAP interpretability for XGBoost
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)