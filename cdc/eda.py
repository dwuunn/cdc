import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('CleanedData.csv')
df = pd.DataFrame(data)

print(data.describe())

encoder = LabelEncoder()
df['readmitted_encoded'] = encoder.fit_transform(df['readmitted'])
df['insulin_encoded'] = encoder.fit_transform(df['insulin'])
df['max_glu_serum_encoded'] = encoder.fit_transform(df['max_glu_serum'])

selected_columns = [
    'readmitted_encoded', 'insulin_encoded', 'max_glu_serum_encoded',
    'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
    'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses'
]

df_selected = df[selected_columns]
correlation_matrix_selected = df_selected.corr()
correlation_matrix_selected

plt.figure(figsize=(15,15))
sns.heatmap(correlation_matrix_selected, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Selected Variables')
plt.xticks(rotation=45, ha='right') 
plt.yticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()