# -*- coding: utf-8 -*-
"""Dev-6-Model-Selection-and-Model-Building-by-Elif-and-Bisma.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Rz0ATDEmBsIxn8z9Mz2ZOSKMYg4Jje1j

**Importing Libraries and Dataset**
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from plotly.graph_objects import Figure
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

pip install scikit-learn==1.2.2

df=pd.read_csv("/content/bank-additional-full.csv",delimiter=';')
df.head(3)

df.shape

df.columns

df.info()

"""**Data Preprocessing**"""

#Dropping unknown values of 'job','marital', 'housing' and 'loan'
df = df[(df['job'] != 'unknown') & (df['marital'] != 'unknown') & (df['loan'] != 'unknown') & (df['housing'] != 'unknown')]

#Imputing 'default' via KNNImputer
df['default'] = df['default'].map({'no': 0, 'yes': 1, 'unknown': np.nan})
columns_for_imputation = df.select_dtypes(include=[np.number]).columns
knn_imputer = KNNImputer(n_neighbors=5)
df[columns_for_imputation] = knn_imputer.fit_transform(df[columns_for_imputation])

plt.figure(figsize=(14, 4))
plt.subplot(121)
sns.histplot(df['default'], kde=True)
plt.xlabel("Default")
plt.ylabel("Frequency")
plt.title("Distribution of Default")

"""**Data Transformation**"""

skew_threshold = 0.5
numeric_columns = ['age','duration', 'campaign', 'pdays',
       'previous','emp.var.rate', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m', 'nr.employed']
skewed_features = [col for col in numeric_columns if df[col].skew() > skew_threshold ]
skewed_features

pt = PowerTransformer()
df_transformed = df.copy()
for col in skewed_features:
    df_transformed[col] = pt.fit_transform(df_transformed[[col]])

df_transformed.duplicated().sum()

df_transformed.drop_duplicates(inplace=True, keep='first')

"""**Data Encoding**"""

numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns
categorical_cols = df_transformed.select_dtypes(include=[object]).columns

# Create DataFrames for numeric and categorical data
df_numeric = df_transformed[numeric_cols]
df_categorical = df_transformed[categorical_cols]

feature_groups = {
  'job': {  # Group similar job types
    'admin.': 'Management & Administration',
    'management': 'Management & Administration',
    'blue-collar': 'Blue-collar & Service',
    'services': 'Blue-collar & Service',
    'professional.course': 'Professional & Technical',
    'technician': 'Professional & Technical',
    'entrepreneur': 'Business & Self-Employed',
    'self-employed': 'Business & Self-Employed',
    'student': 'Non-Active Workforce',
    'unemployed': 'Non-Active Workforce',
    'retired': 'Non-Active Workforce',
    'housemaid': 'Housemaid'
  },
  'education': {  # Group by attainment level
    'basic.4y': 'Basic & Secondary',
    'basic.6y': 'Basic & Secondary',
    'basic.9y': 'Basic & Secondary',
    'high.school': 'Basic & Secondary',
    'professional.course': 'Vocational/Professional & University',
    'university.degree': 'Vocational/Professional & University',
    'illiterate': 'Others',
    'unknown': 'Others'
  },
  'month': {  # Seasonal encoding for month
    'jan': 'Winter',
    'feb': 'Winter',
    'mar': 'Spring',
    'apr': 'Spring',
    'may': 'Spring',
    'jun': 'Summer',
    'jul': 'Summer',
    'aug': 'Summer',
    'sep': 'Fall',
    'oct': 'Fall',
    'nov': 'Fall',
    'dec': 'Winter'
  }
}

for col, group_map in feature_groups.items():
  df_categorical[f'{col}_group'] = df_categorical[col].map(group_map)

df_categorical

df_categorical.drop(feature_groups.keys(), axis=1, inplace=True)

df_categorical

target_mapping = {'yes': 1, 'no': 0}
df_categorical.loc[:, 'y'] = df_categorical['y'].map(target_mapping)
df_categorical.head()

df_categorical.info()
df_categorical['marital'].unique()

encode_df=df_categorical.drop(columns='y')

df_categorical.columns

categorical_cols = ["marital", "housing", "loan", "contact", "day_of_week",
                   "poutcome", "job_group", "education_group", "month_group"]
encoder = OneHotEncoder(handle_unknown='ignore', drop='first')
encoder.fit(encode_df[categorical_cols])
encoded_col_names = encoder.get_feature_names_out(categorical_cols)
encoded_features = encoder.transform(encode_df[categorical_cols])

encoded_features_dense = encoded_features.toarray()
encoded_df = encode_df.assign(**dict(zip(encoded_col_names, encoded_features_dense.T)))

encoded_df=encoded_df.drop(columns=["marital", "housing", "loan", "contact", "day_of_week",
                   "poutcome", "job_group", "education_group", "month_group"],axis=1)

combined_df = pd.concat([df_numeric , df_categorical, encoded_df], axis=1)
combined_df.head(3)

combined_df=combined_df.drop(columns=["marital","housing", "loan", "contact", "day_of_week",
                   "poutcome", "job_group", "education_group", "month_group"],axis=1)

combined_df.info()

combined_df['y'] = pd.to_numeric(combined_df['y'])
combined_df.info()

combined_df['y'].unique()

combined_df.duplicated().sum()

combined_df.drop_duplicates(inplace=True, keep='first')

"""**Feature Selection**"""

fig = Figure()
corr_matrix = combined_df.corr()
corr_matrix_data = corr_matrix.values.tolist()

row_labels = corr_matrix.index.to_numpy()
col_labels = corr_matrix.columns.to_numpy()
fig = Figure(layout=dict(width=800, height=800))
fig.add_trace(dict(
    z=corr_matrix_data,
    x=col_labels,
    y=row_labels.tolist(),
    type='heatmap',
    colorscale='electric'
))
fig.update_layout(
    title='Correlation Heatmap',
    xaxis_title='Features',
    yaxis_title='Features'
)
fig.update_traces(colorbar=dict(dtick=0.2))
fig.show()

df_subset = combined_df[['y', 'default']]

corr_matrix = df_subset.corr(method='spearman')
corr_matrix_data = corr_matrix.values.tolist()
row_labels = corr_matrix.index.to_numpy()
col_labels = corr_matrix.columns.to_numpy()

fig = Figure()
fig.add_trace(dict(
    z=corr_matrix_data,
    x=col_labels,
    y=row_labels.tolist(),
    type='heatmap',
    colorscale='electric'
))

fig.update_layout(
    title='Correlation Heatmap (y vs. default)',
    xaxis_title='Features',
    yaxis_title='Features'
)
fig.update_traces(colorbar=dict(dtick=0.2))
fig.show()

"""As analysed here and in previous Analysis, 'default' does not play important role in predicting target variable so , dropping it too."""

combined_df=combined_df.drop(columns=['day_of_week_mon', 'day_of_week_thu', 'default',
                            'day_of_week_tue', 'day_of_week_wed','month_group_Spring',
                                      'month_group_Summer', 'month_group_Winter'])
##dropping cols in feature selection

combined_df.columns

"""**Checking target variable distribution for balanced dataset**"""

unique_values = combined_df['y'].unique()
value_counts = combined_df['y'].value_counts()

# Check if the order of unique values matches the order of value counts
if (unique_values == value_counts.index).all():
  print("Sorting value counts is not needed. Unique values already match the order of counts.")
else:
  print("Sorting value counts might be helpful for a clearer visualization.")

class_counts = combined_df['y'].value_counts().sort_values(ascending=False)
print('Class Frequencies: \n', class_counts)

majority_class = max(class_counts.values)
minority_class = min(class_counts.values)
imbalance_ratio = majority_class / minority_class
print(f"\nImbalance Ratio (Positive / Negative): {imbalance_ratio:.2f}")


plt.bar(combined_df['y'].unique(), combined_df['y'].value_counts())
plt.title('Distribution of Target Variable (y)')
plt.xlabel('Target Value')
plt.ylabel('Frequency')
plt.show()

"""**Data is highly imbalance, Applying Undersampling and Oversampling for Balance**"""

X = combined_df.drop('y', axis=1)
y = combined_df['y']

rus = RandomUnderSampler(random_state=42)
X_res_us, y_res_us = rus.fit_resample(X, y)

plt.bar(y_res_us.unique(), y_res_us.value_counts())
plt.title('Distribution of Target Variable (y) After Undersampling')
plt.xlabel('Target Value')
plt.ylabel('Frequency')
plt.show()

X_res_smote,y_res_smote=SMOTE().fit_resample(X,y)

plt.bar(y_res_smote.unique(), y_res_smote.value_counts())
plt.title('Distribution of Target Variable (y) After Oversampling')
plt.xlabel('Target Value')
plt.ylabel('Frequency')
plt.show()

"""**Splitting Undersampled data into train and test dataset**"""

X_us_train, X_us_test, y_us_train, y_us_test = train_test_split(X_res_us ,y_res_us, test_size=0.2, random_state=42)

print("Training set shapes:")
print(X_us_train.shape, y_us_train.shape)

print("Testing set shapes:")
print(X_us_test.shape, y_us_test.shape)

"""**Model Selection**

**Models Performance on Undersampled dataset**

**1. Logistic Regression**
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_curve, auc, confusion_matrix

lg = LogisticRegression()

lg.fit(X_us_train, y_us_train)

lg_y_pred = lg.predict(X_us_test)

lg_accuracy = accuracy_score(y_us_test, lg_y_pred)
lg_precision = precision_score(y_us_test, lg_y_pred)
lg_recall = recall_score(y_us_test, lg_y_pred)
lg_f1 = f1_score(y_us_test, lg_y_pred)

fpr_lg, tpr_lg, _ = roc_curve(y_us_test, lg_y_pred)
lg_roc_auc = auc(fpr_lg, tpr_lg)

lg_confusion_matrix = confusion_matrix(y_us_test, lg_y_pred)
print("Logistic Regression Performance for Undersampled dataset:")
print(f"Accuracy: {lg_accuracy:.4f}")
print(f"Precision: {lg_precision:.4f}")
print(f"Recall: {lg_recall:.4f}")
print(f"F1-score: {lg_f1:.4f}")

print(f"ROC: {lg_roc_auc:.4f}")
plt.figure(figsize=(8, 6))
plt.plot(fpr_lg, tpr_lg, label='Log Reg (AUC = %0.4f)' % lg_roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='No Skill')
plt.title('ROC Curve (Logistic Regression)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print(f"Confusion Matrix:")
plt.figure(figsize=(14,7))
sns.heatmap(lg_confusion_matrix, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

"""**2. SVM**"""

from sklearn.svm import SVC
svm = SVC()

svm.fit(X_us_train, y_us_train)

svm_y_pred = svm.predict(X_us_test)

svm_accuracy = accuracy_score(y_us_test, svm_y_pred)
svm_precision = precision_score(y_us_test, svm_y_pred)
svm_recall = recall_score(y_us_test, svm_y_pred)
svm_f1 = f1_score(y_us_test, svm_y_pred)

fpr_svm, tpr_svm, _ = roc_curve(y_us_test, svm_y_pred)
svm_roc_auc = auc(fpr_svm, tpr_svm)

svm_confusion_matrix = confusion_matrix(y_us_test, svm_y_pred)

print("\nSVM Model Performance for Undersampled Data:")
print(f"Accuracy: {svm_accuracy:.4f}")
print(f"Precision: {svm_precision:.4f}")
print(f"Recall: {svm_recall:.4f}")
print(f"F1-score: {svm_f1:.4f}")

print(f"ROC: {svm_roc_auc:.4f}")
plt.figure(figsize=(8, 6))
plt.plot(fpr_svm, tpr_svm, label='SVM (AUC = %0.4f)' % svm_roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='No Skill')
plt.title('ROC Curve (SVM)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print(f"Confusion Matrix:")
plt.figure(figsize=(14,7))
sns.heatmap(svm_confusion_matrix, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

"""**3. Decision Tree**"""

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_us_train, y_us_train)

dtree_y_pred = dtree.predict(X_us_test)

dtree_accuracy = accuracy_score(y_us_test, dtree_y_pred)
dtree_precision = precision_score(y_us_test, dtree_y_pred)
dtree_recall = recall_score(y_us_test, dtree_y_pred)
dtree_f1 = f1_score(y_us_test, dtree_y_pred)


fpr_dtree, tpr_dtree, _ = roc_curve(y_us_test, dtree_y_pred)
dtree_roc_auc = auc(fpr_dtree, tpr_dtree)

dtree_confusion_matrix = confusion_matrix(y_us_test, dtree_y_pred)

print("Decision Tree Model Performance for Undersampled dataset:")
print(f"Accuracy: {dtree_accuracy:.4f}")
print(f"Precision: {dtree_precision:.4f}")
print(f"Recall: {dtree_recall:.4f}")
print(f"F1-score: {dtree_f1:.4f}")

print(f"ROC: {dtree_roc_auc:.4f}")
plt.figure(figsize=(8, 6))
plt.plot(fpr_dtree, tpr_dtree, label='Dtree (AUC = %0.4f)' % dtree_roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='No Skill')
plt.title('ROC Curve (Dtree)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print(f"Confusion Matrix:")
plt.figure(figsize=(14,7))
sns.heatmap(dtree_confusion_matrix, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

"""**4. Random Forest**"""

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)

rf.fit(X_us_train, y_us_train)

rf_y_pred = rf.predict(X_us_test)

rf_accuracy = accuracy_score(y_us_test, rf_y_pred)
rf_precision = precision_score(y_us_test, rf_y_pred)
rf_recall = recall_score(y_us_test, rf_y_pred)
rf_f1 = f1_score(y_us_test, rf_y_pred)

fpr_rf, tpr_rf, _ = roc_curve(y_us_test, rf_y_pred)
rf_roc_auc = auc(fpr_rf, tpr_rf)

rf_confusion_matrix = confusion_matrix(y_us_test, rf_y_pred)

print("Random Forest Model Performance:")
print(f"Accuracy: {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall: {rf_recall:.4f}")
print(f"F1-score: {rf_f1:.4f}")

print(f"ROC: {rf_roc_auc:.4f}")
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label='RF (AUC = %0.4f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='No Skill')
plt.title('ROC Curve (RF)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print(f"Confusion Matrix:")
plt.figure(figsize=(14,7))
sns.heatmap(rf_confusion_matrix, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

"""**5. XGBoost**"""

from xgboost import XGBClassifier
xgb= XGBClassifier(objective='binary:logistic', random_state=42)

xgb.fit(X_us_train, y_us_train)

xgb_y_pred = xgb.predict(X_us_test)

xgb_accuracy = accuracy_score(y_us_test, xgb_y_pred)
xgb_precision = precision_score(y_us_test, xgb_y_pred)
xgb_recall = recall_score(y_us_test, xgb_y_pred)
xgb_f1 = f1_score(y_us_test, xgb_y_pred)

fpr_xgb, tpr_xgb, _ = roc_curve(y_us_test, xgb_y_pred)
xgb_roc_auc = auc(fpr_xgb, tpr_xgb)

xgb_confusion_matrix = confusion_matrix(y_us_test, xgb_y_pred)


print("XGBoost Model Performance:")
print(f"Accuracy: {xgb_accuracy:.4f}")
print(f"Precision: {xgb_precision:.4f}")
print(f"Recall: {xgb_recall:.4f}")
print(f"F1-score: {xgb_f1:.4f}")
print(f"ROC: {xgb_roc_auc:.4f}")
plt.figure(figsize=(8, 6))
plt.plot(fpr_xgb, tpr_xgb, label='XGB(AUC = %0.4f)' % xgb_roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='No Skill')
plt.title('ROC Curve (XGB)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print(f"Confusion Matrix:")
plt.figure(figsize=(14,7))
sns.heatmap(xgb_confusion_matrix, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

from lightgbm import LGBMClassifier

lgb = LGBMClassifier(objective='binary', random_state=42)

lgb.fit(X_us_train, y_us_train)

lgb_y_pred = lgb.predict(X_us_test)

lgb_accuracy = accuracy_score(y_us_test, lgb_y_pred)
lgb_precision = precision_score(y_us_test, lgb_y_pred)
lgb_recall = recall_score(y_us_test, lgb_y_pred)
lgb_f1 = f1_score(y_us_test, lgb_y_pred)


fpr_lgb, tpr_lgb, _ = roc_curve(y_us_test, lgb_y_pred)
lgb_roc_auc = auc(fpr_lgb, tpr_lgb)

lgb_confusion_matrix = confusion_matrix(y_us_test, lgb_y_pred)

print("XGBoost Model Performance:")
print(f"Accuracy: {lgb_accuracy:.4f}")
print(f"Precision: {lgb_precision:.4f}")
print(f"Recall: {lgb_recall:.4f}")
print(f"F1-score: {lgb_f1:.4f}")
print(f"ROC: {lgb_roc_auc:.4f}")
plt.figure(figsize=(8, 6))
plt.plot(fpr_lgb, tpr_lgb, label='LightGBM (AUC = %0.4f)' % lgb_roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='No Skill')
plt.title('ROC Curve (LightGBM)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print(f"Confusion Matrix:")
plt.figure(figsize=(14,7))
sns.heatmap(lgb_confusion_matrix, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

"""**Models Performance on SMOTE (OverSampled) Dataset**"""

X_smote_train, X_smote_test, y_smote_train, y_smote_test = train_test_split(X_res_smote ,y_res_smote, test_size=0.2, random_state=42)

print("Training set shapes:")
print(X_smote_train.shape, y_smote_train.shape)

print("Testing set shapes:")
print(X_smote_test.shape, y_smote_test.shape)

"""**1. Logistic Regression**"""

lg.fit(X_smote_train, y_smote_train)

lg_y_pred_smote = lg.predict(X_smote_test)

lg_accuracy_smote = accuracy_score(y_smote_test, lg_y_pred_smote)
lg_precision_smote = precision_score(y_smote_test, lg_y_pred_smote)
lg_recall_smote = recall_score(y_smote_test, lg_y_pred_smote)
lg_f1_smote = f1_score(y_smote_test, lg_y_pred_smote)

fpr_lg_smote, tpr_lg_smote, _smote = roc_curve(y_smote_test, lg_y_pred_smote)
lg_roc_auc_smote = auc(fpr_lg_smote, tpr_lg_smote)

lg_confusion_matrix_smote = confusion_matrix(y_smote_test, lg_y_pred_smote)
print("Logistic Regression Performance for Oversampled (SMOTE) dataset:")
print(f"Accuracy: {lg_accuracy_smote:.4f}")
print(f"Precision: {lg_precision_smote:.4f}")
print(f"Recall: {lg_recall_smote:.4f}")
print(f"F1-score: {lg_f1_smote:.4f}")

print(f"ROC: {lg_roc_auc_smote:.4f}")
plt.figure(figsize=(8, 6))
plt.plot(fpr_lg_smote, tpr_lg_smote, label='Log Reg (AUC = %0.4f)' % lg_roc_auc_smote)
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='No Skill')
plt.title('ROC Curve (Logistic Regression)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print(f"Confusion Matrix:")
plt.figure(figsize=(14,7))
sns.heatmap(lg_confusion_matrix_smote, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

svm.fit(X_smote_train, y_smote_train)

svm_y_pred_smote = svm.predict(X_smote_test)

svm_accuracy_smote = accuracy_score(y_smote_test, svm_y_pred_smote)
svm_precision_smote = precision_score(y_smote_test, svm_y_pred_smote)
svm_recall_smote = recall_score(y_smote_test, svm_y_pred_smote)
svm_f1_smote = f1_score(y_smote_test, svm_y_pred_smote)

fpr_svm_smote, tpr_svm_smote, _smote = roc_curve(y_smote_test, svm_y_pred_smote)
svm_roc_auc_smote = auc(fpr_svm_smote, tpr_svm_smote)

svm_confusion_matrix_smote = confusion_matrix(y_smote_test, svm_y_pred_smote)
print("Logistic Regression Performance for Oversampled (SMOTE) dataset:")
print(f"Accuracy: {svm_accuracy_smote:.4f}")
print(f"Precision: {svm_precision_smote:.4f}")
print(f"Recall: {svm_recall_smote:.4f}")
print(f"F1-score: {svm_f1_smote:.4f}")

print(f"ROC: {svm_roc_auc_smote:.4f}")
plt.figure(figsize=(8, 6))
plt.plot(fpr_svm_smote, tpr_svm_smote, label='SVM(AUC = %0.4f)' % svm_roc_auc_smote)
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='No Skill')
plt.title('ROC Curve (SVM)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print(f"Confusion Matrix:")
plt.figure(figsize=(14,7))
sns.heatmap(svm_confusion_matrix_smote, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

dtree.fit(X_smote_train, y_smote_train)

dtree_y_pred_smote = dtree.predict(X_smote_test)


dtree_accuracy_smote = accuracy_score(y_smote_test, dtree_y_pred_smote)
dtree_precision_smote = precision_score(y_smote_test, dtree_y_pred_smote)
dtree_recall_smote = recall_score(y_smote_test, dtree_y_pred_smote)
dtree_f1_smote = f1_score(y_smote_test, dtree_y_pred_smote)

fpr_dtree_smote, tpr_dtree_smote, _smote = roc_curve(y_smote_test, dtree_y_pred_smote)
dtree_roc_auc_smote = auc(fpr_dtree_smote, tpr_dtree_smote)

dtree_confusion_matrix_smote = confusion_matrix(y_smote_test, dtree_y_pred_smote)

print("Decision Tree Performance for Oversampled (SMOTE) dataset:")
print(f"Accuracy: {dtree_accuracy_smote:.4f}")
print(f"Precision: {dtree_precision_smote:.4f}")
print(f"Recall: {dtree_recall_smote:.4f}")
print(f"F1-score: {dtree_f1_smote:.4f}")

print(f"ROC: {dtree_roc_auc_smote:.4f}")
plt.figure(figsize=(8, 6))
plt.plot(fpr_dtree_smote, tpr_dtree_smote, label='Decision Tree (AUC = %0.4f)' % dtree_roc_auc_smote)
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='No Skill')
plt.title('ROC Curve (Decision Tree)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print(f"Confusion Matrix:")
plt.figure(figsize=(14,7))
sns.heatmap(dtree_confusion_matrix_smote, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

rf.fit(X_smote_train, y_smote_train)

rf_y_pred_smote = rf.predict(X_smote_test)

rf_accuracy_smote = accuracy_score(y_smote_test, rf_y_pred_smote)
rf_precision_smote = precision_score(y_smote_test, rf_y_pred_smote)
rf_recall_smote = recall_score(y_smote_test, rf_y_pred_smote)
rf_f1_smote = f1_score(y_smote_test, rf_y_pred_smote)

fpr_rf_smote, tpr_rf_smote, _smote = roc_curve(y_smote_test, rf_y_pred_smote)
rf_roc_auc_smote = auc(fpr_rf_smote, tpr_rf_smote)

rf_confusion_matrix_smote = confusion_matrix(y_smote_test, rf_y_pred_smote)

print("Random Forest Performance for Oversampled (SMOTE) dataset:")
print(f"Accuracy: {rf_accuracy_smote:.4f}")
print(f"Precision: {rf_precision_smote:.4f}")
print(f"Recall: {rf_recall_smote:.4f}")
print(f"F1-score: {rf_f1_smote:.4f}")

print(f"ROC: {rf_roc_auc_smote:.4f}")
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf_smote, tpr_rf_smote, label='Random Forest (AUC = %0.4f)' % rf_roc_auc_smote)
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='No Skill')
plt.title('ROC Curve (Random Forest)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print(f"Confusion Matrix:")
plt.figure(figsize=(14,7))
sns.heatmap(rf_confusion_matrix_smote, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

xgb.fit(X_smote_train, y_smote_train)

xgb_y_pred_smote = xgb.predict(X_smote_test)

xgb_accuracy_smote = accuracy_score(y_smote_test, xgb_y_pred_smote)
xgb_precision_smote = precision_score(y_smote_test, xgb_y_pred_smote)
xgb_recall_smote = recall_score(y_smote_test, xgb_y_pred_smote)
xgb_f1_smote = f1_score(y_smote_test, xgb_y_pred_smote)

fpr_xgb_smote, tpr_xgb_smote, _smote = roc_curve(y_smote_test, xgb_y_pred_smote)
xgb_roc_auc_smote = auc(fpr_xgb_smote, tpr_xgb_smote)

xgb_confusion_matrix_smote = confusion_matrix(y_smote_test, xgb_y_pred_smote)

print("XGBoost Performance for Oversampled (SMOTE) dataset:")
print(f"Accuracy: {xgb_accuracy_smote:.4f}")
print(f"Precision: {xgb_precision_smote:.4f}")
print(f"Recall: {xgb_recall_smote:.4f}")
print(f"F1-score: {xgb_f1_smote:.4f}")

print(f"ROC: {xgb_roc_auc_smote:.4f}")
plt.figure(figsize=(8, 6))
plt.plot(fpr_xgb_smote, tpr_xgb_smote, label='XGBoost (AUC = %0.4f)' % xgb_roc_auc_smote)
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='No Skill')
plt.title('ROC Curve (XGBoost)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print(f"Confusion Matrix:")
plt.figure(figsize=(14,7))
sns.heatmap(xgb_confusion_matrix_smote, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

lgb.fit(X_smote_train, y_smote_train)

lgb_y_pred_smote = lgb.predict(X_smote_test)

lgb_accuracy_smote = accuracy_score(y_smote_test, lgb_y_pred_smote)
lgb_precision_smote = precision_score(y_smote_test, lgb_y_pred_smote)
lgb_recall_smote = recall_score(y_smote_test, lgb_y_pred_smote)
lgb_f1_smote = f1_score(y_smote_test, lgb_y_pred_smote)

fpr_lgb_smote, tpr_lgb_smote, _smote = roc_curve(y_smote_test, lgb_y_pred_smote)
lgb_roc_auc_smote = auc(fpr_lgb_smote, tpr_lgb_smote)

lgb_confusion_matrix_smote = confusion_matrix(y_smote_test, lgb_y_pred_smote)

print("LightGBM Performance for Oversampled (SMOTE) dataset:")
print(f"Accuracy: {lgb_accuracy_smote:.4f}")
print(f"Precision: {lgb_precision_smote:.4f}")
print(f"Recall: {lgb_recall_smote:.4f}")
print(f"F1-score: {lgb_f1_smote:.4f}")

print(f"ROC: {lgb_roc_auc_smote:.4f}")
plt.figure(figsize=(8, 6))
plt.plot(fpr_lgb_smote, tpr_lgb_smote, label='LightGBM (AUC = %0.4f)' % lgb_roc_auc_smote)
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='No Skill')
plt.title('ROC Curve (LightGBM)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print(f"Confusion Matrix:")
plt.figure(figsize=(14,7))
sns.heatmap(lgb_confusion_matrix_smote, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

result_table_us=pd.DataFrame({'Models':['LR','SVM','DT','RF','XGB','LGB'],
                              'Accuracy':[lg_accuracy,svm_accuracy,dtree_accuracy,rf_accuracy,xgb_accuracy,lgb_accuracy],
                              'Precision':[lg_precision,svm_precision,dtree_precision,rf_precision,xgb_precision,lgb_precision],
                              'Recall':[lg_recall,svm_recall,dtree_recall,rf_recall,xgb_recall,lgb_recall],
                              'F1':[lg_f1,svm_f1,dtree_f1,rf_f1,xgb_f1,lgb_f1],
                              'ROC-Curve':[lg_roc_auc,svm_roc_auc,dtree_roc_auc,rf_roc_auc,xgb_roc_auc,lgb_roc_auc]})

result_table_smote=pd.DataFrame({'Models':['LR','SVM','DT','RF','XGB','LGB'],
                              'Accuracy':[lg_accuracy_smote,svm_accuracy_smote,dtree_accuracy_smote,rf_accuracy_smote,xgb_accuracy_smote,lgb_accuracy_smote],
                              'Precision':[lg_precision_smote,svm_precision_smote,dtree_precision_smote,rf_precision_smote,xgb_precision_smote,lgb_precision_smote],
                              'Recall':[lg_recall_smote,svm_recall_smote,dtree_recall_smote,rf_recall_smote,xgb_recall_smote,lgb_recall_smote],
                              'F1':[lg_f1_smote,svm_f1_smote,dtree_f1_smote,rf_f1_smote,xgb_f1_smote,lgb_f1_smote],
                              'ROC-Curve':[lg_roc_auc_smote,svm_roc_auc_smote,dtree_roc_auc_smote,rf_roc_auc_smote,xgb_roc_auc_smote,lgb_roc_auc_smote]})

result_table_us.sort_values(by='ROC-Curve',ascending=False).style.background_gradient(cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True))

result_table_smote.sort_values(by='ROC-Curve',ascending=False).style.background_gradient(cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True))

"""**Dataset balanced via SMOTE performed very well using Random Forest Algorithm.**
**Investigating further for selection of most suitable Model.**

**Model Evaluation**

*Tuning top three best performing models*

**1. Random Forest**
"""

rf = RandomForestClassifier(n_estimators=200, max_depth=25, random_state=42)


rf.fit(X_smote_train, y_smote_train)

rf_y_pred_smote = rf.predict(X_smote_test)

rf_accuracy_smote = accuracy_score(y_smote_test, rf_y_pred_smote)
rf_precision_smote = precision_score(y_smote_test, rf_y_pred_smote)
rf_recall_smote = recall_score(y_smote_test, rf_y_pred_smote)
rf_f1_smote = f1_score(y_smote_test, rf_y_pred_smote)

fpr_rf_smote, tpr_rf_smote, _smote = roc_curve(y_smote_test, rf_y_pred_smote)
rf_roc_auc_smote = auc(fpr_rf_smote, tpr_rf_smote)

print("Random Forest Performance for Oversampled (SMOTE) dataset:")
print(f"Accuracy: {rf_accuracy_smote:.4f}")
print(f"Precision: {rf_precision_smote:.4f}")
print(f"Recall: {rf_recall_smote:.4f}")
print(f"F1-score: {rf_f1_smote:.4f}")

print(f"ROC: {rf_roc_auc_smote:.4f}")

lgb = LGBMClassifier(objective='binary', random_state=42, n_estimators=400, max_depth=25, learning_rate=0.1)

lgb.fit(X_smote_train, y_smote_train)

lgb_y_pred_smote = lgb.predict(X_smote_test)

lgb_accuracy_smote = accuracy_score(y_smote_test, lgb_y_pred_smote)
lgb_precision_smote = precision_score(y_smote_test, lgb_y_pred_smote)
lgb_recall_smote = recall_score(y_smote_test, lgb_y_pred_smote)
lgb_f1_smote = f1_score(y_smote_test, lgb_y_pred_smote)

fpr_lgb_smote, tpr_lgb_smote, _smote = roc_curve(y_smote_test, lgb_y_pred_smote)
lgb_roc_auc_smote = auc(fpr_lgb_smote, tpr_lgb_smote)

lgb_confusion_matrix_smote = confusion_matrix(y_smote_test, lgb_y_pred_smote)

print("LightGBM Performance for Oversampled (SMOTE) dataset:")
print(f"Accuracy: {lgb_accuracy_smote:.4f}")
print(f"Precision: {lgb_precision_smote:.4f}")
print(f"Recall: {lgb_recall_smote:.4f}")
print(f"F1-score: {lgb_f1_smote:.4f}")

print(f"ROC: {lgb_roc_auc_smote:.4f}")

xgb= XGBClassifier(objective='binary:logistic', random_state=42, n_estimators=300, max_depth=15, learning_rate=0.1)

xgb.fit(X_smote_train, y_smote_train)

xgb_y_pred_smote = xgb.predict(X_smote_test)

xgb_accuracy_smote = accuracy_score(y_smote_test, xgb_y_pred_smote)
xgb_precision_smote = precision_score(y_smote_test, xgb_y_pred_smote)
xgb_recall_smote = recall_score(y_smote_test, xgb_y_pred_smote)
xgb_f1_smote = f1_score(y_smote_test, xgb_y_pred_smote)

fpr_xgb_smote, tpr_xgb_smote, _smote = roc_curve(y_smote_test, xgb_y_pred_smote)
xgb_roc_auc_smote = auc(fpr_xgb_smote, tpr_xgb_smote)
xgb_confusion_matrix_smote = confusion_matrix(y_smote_test, xgb_y_pred_smote)

print("XGBoost Performance for Oversampled (SMOTE) dataset:")
print(f"Accuracy: {xgb_accuracy_smote:.4f}")
print(f"Precision: {xgb_precision_smote:.4f}")
print(f"Recall: {xgb_recall_smote:.4f}")
print(f"F1-score: {xgb_f1_smote:.4f}")
print(f"ROC: {xgb_roc_auc_smote:.4f}")

"""**Even After Models' Evaluation, 1st Random Forest via SMOTE performed the best, So using it as final model.**

**Checking if RF is overfitting or not.**
"""

X1_train, X1_val, y1_train, y1_val = train_test_split(X_res_smote ,y_res_smote, test_size=0.2)

rf1 = RandomForestClassifier(random_state=42)
rf1.fit(X1_train, y1_train)

train_score_before = rf1.score(X1_train, y1_train)
val_score_before = rf1.score(X1_val, y1_val)


print("Training Score before:", train_score_before)
print("Validation Score before:", val_score_before)

#Commenting out this code as it takes very long to run.
'''
from sklearn.model_selection import GridSearchCV, StratifiedKFold

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

param_grid = {
    'max_depth': [3, 5, 8, 10, 12, 15],
    'n_estimators': [100, 200, 300, 400],
    'max_features': ['sqrt', 'log2']
}


rf_clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=skf, scoring='f1_macro', return_train_score=False, verbose=2)
rf_clf.fit(X1_train, y1_train)


best_rf_model = rf_clf.best_estimator_
predictions = best_rf_model.predict(X1_val)

print("Best hyperparameters:", rf_clf.best_params_)
'''

X1_train, X1_val, y1_train, y1_val = train_test_split(X_res_smote ,y_res_smote, test_size=0.2)

rf_best = RandomForestClassifier(random_state=42, max_depth=11, max_features='sqrt', n_estimators=200)
rf_best.fit(X1_train, y1_train)

train_score_after = rf_best.score(X1_train, y1_train)
val_score_after = rf_best.score(X1_val, y1_val)


print("Training Score After:", train_score_after)
print("Validation Score After:", val_score_after)

"""*Now that the difference btw Validation Score and Training Score is relatively less, the new scores provide a positive trend. This indicates that the model is likely generalizing better and learning patterns that apply to unseen data.*"""

train_score = 0.9334
val_score = 0.9232
scores = [train_score, val_score]
labels = ['Train Score', 'Test Score']

plt.figure(figsize=(3, 3))
plt.bar(labels, scores,color='lightgreen')
plt.title('Train vs. Test Scores of Random Forest')
plt.xlabel('Score Type')
plt.ylabel('Score')
plt.show()

"""**Finalizing the Model**"""

rf_best = RandomForestClassifier(random_state=42, max_depth=11, max_features='sqrt', n_estimators=200)
rf_best.fit(X_smote_train, y_smote_train)

rf_best.predict(X_smote_test)

import pickle
filename = 'bank_model.pkl'
pickle.dump(rf_best, open(filename, 'wb'))