import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dtreeviz import model

from xgboost import plot_tree

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

raw_data = pd.read_csv('data/top_breaches.csv')

raw_data = raw_data.iloc[1:]
raw_data = raw_data[['sector', 'method', 'data sensitivity', 'date', 'records lost']]

raw_data['records lost'] = raw_data['records lost'].str.replace(',', '').str.replace(';', '').astype(int)

label_encoder = LabelEncoder()
for column in ['sector', 'method', 'data sensitivity', 'date']:
    raw_data[column] = label_encoder.fit_transform(raw_data[column])

X = raw_data[['sector', 'method', 'data sensitivity', 'date']]
y = raw_data['records lost']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)

xgb_regressor = XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_regressor.fit(X_train, y_train)
viz_model = model(
    xgb_regressor.get_booster(),
    X_train=X_train,
    y_train=y_train,
    feature_names=['sector', 'method', 'data sensitivity', 'date'],
    target_name="records lost",
    tree_index=2  
)


plt.figure(figsize=(30, 20))  
plot_tree(xgb_regressor, num_trees=2) 
plt.show()

viz_model.view()
y_pred = xgb_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# feature importance
importance = xgb_regressor.feature_importances_
print("Feature Importances:", importance)

# feature importance visualization
plt.bar(range(len(importance)), importance)
plt.xticks(range(len(importance)), X.columns, rotation=45)
plt.title("Feature Importance")
plt.show()
