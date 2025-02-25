import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

raw_data = pd.read_csv('top_breaches.csv')

raw_data = raw_data.iloc[1:]
raw_data = raw_data[['sector', 'method', 'data sensitivity', 'date', 'records lost']]

raw_data['records lost'] = raw_data['records lost'].str.replace(',', '').str.replace(';', '').astype(int)

label_encoder = LabelEncoder()
for column in ['sector', 'method', 'data sensitivity', 'date']:
    raw_data[column] = label_encoder.fit_transform(raw_data[column])

X = raw_data[['sector', 'method', 'data sensitivity', 'date']]
y = raw_data['records lost']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)

model = XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# feature importance
importance = model.feature_importances_
print("Feature Importances:", importance)

# feature importance visualization
plt.bar(range(len(importance)), importance)
plt.xticks(range(len(importance)), X.columns, rotation=45)
plt.title("Feature Importance")
plt.show()
