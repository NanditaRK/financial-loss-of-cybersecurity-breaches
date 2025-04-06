import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

raw_data = pd.read_csv('data/vcdb.csv', low_memory=False)

# these wwere the categories we deemed most important throught 
keywords = [
    'security_incident',
    'impact.overall_rating',
    'cost_corrective_action',
    'asset.variety',
    'asset.accessibility',
    'asset.ownership',
    'impact.loss.variety (Legal and regulatory)',
    'response_and_recovery',
    'actor.x.motive',
    'actor.external.variety',
    'timeline.unit',
    'timeline.duration',
    'targeted',
    'victim.employee_count',
    'attribute.confidentiality.data_disclosure',
    'attribute.confidentiality.data.variety',
]

filtered_columns = [col for col in raw_data.columns if any(keyword in col for keyword in keywords)]
filtered_data = raw_data[filtered_columns]

impact_columns = filtered_data.filter(like='impact.overall_rating')
if "impact.overall_rating.Unknown" in impact_columns.columns:
    impact_columns = impact_columns.drop(columns="impact.overall_rating.Unknown")

mask = impact_columns.any(axis=1)

final_data = filtered_data[mask]

final_data['impact_category'] = impact_columns.idxmax(axis=1)
final_data['impact_category'] = final_data['impact_category'].str.replace('impact.overall_rating.', '')

X = final_data.drop(columns=impact_columns.columns.tolist() + ['impact_category'])
y = final_data['impact_category']

X = X.astype(int)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=87, stratify=y_encoded)

print("Class distribution before SMOTE:", np.bincount(y_train))

min_samples_class = min(np.bincount(y_train))

n_neighbors = min(min_samples_class - 1, 5) 

smote = SMOTE(sampling_strategy='auto', k_neighbors=n_neighbors, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Class distribution after SMOTE:", np.bincount(y_train_resampled))

xgb_model = XGBClassifier(
    objective='multi:softmax',
    num_class=len(label_encoder.classes_),
    eval_metric='mlogloss',
    random_state=42
)

param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 6, 10],
    'n_estimators': [50, 100, 200],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}


grid_search = GridSearchCV(xgb_model, param_grid, scoring='accuracy', cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)

print("Best parameters from grid search:", grid_search.best_params_)

best_model = grid_search.best_estimator_

best_model.fit(X_train_resampled, y_train_resampled)

train_accuracy = best_model.score(X_train_resampled, y_train_resampled)
test_accuracy = best_model.score(X_test, y_test)
y_pred = best_model.predict(X_test)

y_pred_labels = label_encoder.inverse_transform(y_pred)
y_test_labels = label_encoder.inverse_transform(y_test)

print("Cross Validations Scores: ", cross_val_score(best_model, X_test, y_test, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)) )
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print("\nConfusion Matrix:\n", confusion_matrix(y_test_labels, y_pred_labels, labels=label_encoder.classes_))
print("\nClassification Report:\n", classification_report(y_test_labels, y_pred_labels, target_names=label_encoder.classes_))

feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
plt.title('Top 10 Feature Importances (XGBoost)')
plt.tight_layout()
plt.show()



import shap

# Visualize SHAP values
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=True)
for class_index in range(len(label_encoder.classes_)):
    print(f"\nGenerating SHAP summary plot for class: {label_encoder.classes_[class_index]}")
    shap.summary_plot(
        shap_values[class_index],
        X_test,
        plot_type="bar",
        show=True
    )
