import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import mutual_info_classif

# Sample data
data = pd.read_csv('bank-additional-full.csv')

# Group and encode categorical features
def group_job(job):
    if job in ['admin.', 'management']:
        return 'Management & Administration'
    elif job in ['blue-collar', 'services']:
        return 'Blue-collar & Service'
    elif job in ['professional.course', 'technician']:
        return 'Professional'
    else:
        return 'Others'

data['job_grouped'] = data['job'].apply(group_job)

def group_education(edu):
    if edu in ['basic.4y', 'basic.6y', 'basic.9y']:
        return 'Basic Education'
    elif edu == 'high.school':
        return 'Secondary Education'
    elif edu == 'professional.course':
        return 'Vocational/Professional'
    elif edu == 'university.degree':
        return 'University'
    else:
        return 'Unknown'

data['education_grouped'] = data['education'].apply(group_education)

def encode_season(month):
    if month in ['jan', 'feb', 'mar']:
        return 'Spring'
    elif month in ['apr', 'may', 'jun']:
        return 'Summer'
    elif month in ['jul', 'aug', 'sep']:
        return 'Fall'
    else:
        return 'Winter'

data['season'] = data['month'].apply(encode_season)

# One-Hot Encode the grouped features
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(data[['job_grouped', 'marital', 'education_grouped', 'contact', 'season', 'day_of_week']])

# Create a DataFrame from the encoded features
encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out())

# Combine encoded features with the rest of the data
data = data.join(encoded_df).drop(columns=['job', 'education', 'month', 'day_of_week', 'job_grouped', 'education_grouped', 'season'])

# Feature selection
# Numerical features: Drop highly correlated features
corr_matrix = data.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
data = data.drop(columns=to_drop)

# Categorical features: WoE and IV
# For demonstration, we'll use mutual information as a proxy for IV
X = data.drop(columns=['target'])
y = data['target']
mi = mutual_info_classif(X, y)
mi_series = pd.Series(mi, index=X.columns)
significant_features = mi_series[mi_series > 0.1].index
data = data[significant_features]

# Now ready for model training
