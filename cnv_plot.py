import umap
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, f1_score, make_scorer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import joblib
from io import StringIO
import requests
import seaborn as sns  # 더 나은 시각화를 위해 seaborn 추가


# 1. 데이터 로드 및 병합

normal_data = pd.read_csv("cnv_normal.csv")
normal_data['Cancer'] = 0
cancer_data = pd.read_csv("cnv_cancer.csv")
cancer_data['Cancer'] = 1
all_data = pd.concat([normal_data, cancer_data], ignore_index=True)
# --- 데이터 로드 끝 ---

all_data = pd.concat([normal_data, cancer_data], ignore_index=True)

# 2. Chromosome 조각 ID 생성
bin_size = 1000000
def create_chromosome_segments(df, bin_size):
    segments = []
    for chrom in df['Chromosome'].unique():
        chrom_df = df[df['Chromosome'] == chrom]
        for start in range(chrom_df['Start'].min(), chrom_df['End'].max(), bin_size):
            end = start + bin_size
            segments.append({'Chromosome': chrom, 'Start': start, 'End': end})
    return pd.DataFrame(segments)

segment_df = create_chromosome_segments(all_data, bin_size)

# 3. 데이터 병합
merged_data = pd.merge(all_data, segment_df, on='Chromosome', how='inner')
merged_data = merged_data[(merged_data['Start_x'] < merged_data['End_y']) & (merged_data['End_x'] > merged_data['Start_y'])]
merged_data['Chromosome_segment'] = merged_data['Chromosome'].astype(str) + '_' + merged_data['Start_y'].astype(str) + '_' + merged_data['End_y'].astype(str) + f'_{bin_size/1000000}Mb'

# 4~6. 가중 평균 계산 및 준비 (동일)
merged_data['Segment_Length'] = merged_data['End_y'] - merged_data['Start_y']
merged_data['Segment_Length'] = merged_data['Segment_Length'].fillna(0)
merged_data['Num_Probes'] = merged_data['Num_Probes'].fillna(0)
merged_data['Segment_Mean'] = merged_data['Segment_Mean'].fillna(merged_data['Segment_Mean'].median())
merged_data.loc[merged_data['Segment_Length'] < 0, 'Segment_Length'] = 0
merged_data.loc[merged_data['Num_Probes'] < 0, 'Num_Probes'] = 0

def safe_minmax_scale(series):
    if series.min() == series.max():  return pd.Series([0] * len(series), index=series.index)
    return (series - series.min()) / (series.max() - series.min())
merged_data['Segment_Length_Norm'] = safe_minmax_scale(merged_data['Segment_Length'])
merged_data['Num_Probes_Norm'] = safe_minmax_scale(merged_data['Num_Probes'])
merged_data['Combined_Weight'] = (merged_data['Segment_Length_Norm'] + merged_data['Num_Probes_Norm']) / 2
merged_data['Combined_Weight'] = merged_data['Combined_Weight'].fillna(0).clip(lower=0)

def weighted_average(df, values, weights):
    try: return np.average(df[values], weights=df[weights])
    except ZeroDivisionError: return 0
    except Exception as e: return 0 if df[weights].isnull().any() or (df[weights] <= 0).any() else 0

# 8. 가중 평균 계산 (ID 포함!)
weighted_mean = merged_data.groupby(['ID', 'Chromosome_segment', 'Cancer']).apply(weighted_average, values='Segment_Mean', weights='Combined_Weight', include_groups=False).reset_index(name='Segment_Mean_Weighted')

# 9. 피벗 테이블 (ID 사용!)
wide_data = weighted_mean.pivot_table(index='ID', columns='Chromosome_segment', values='Segment_Mean_Weighted').reset_index()

# wide_data에 Cancer 열 추가.
# 방법 1 (ID, Cancer 쌍이 unique한 경우)
cancer_info = merged_data.drop_duplicates(subset=['ID'])[['ID','Cancer']]
wide_data = pd.merge(wide_data, cancer_info, on='ID', how='left')


# --- X, y 생성 (wide_data 기반) ---
# X: wide_data에서 'ID', 'Cancer' 열을 제외한 모든 열
X = wide_data.drop(['ID','Cancer'], axis=1,errors = 'ignore')
X = X.fillna(X.mean()) # 결측치 처리

# y:  wide_data에서 Cancer 열 사용
y = wide_data['Cancer']

# --- 데이터 불균형 확인 ---
print("Class distribution before resampling:\n", y.value_counts(normalize=True))

# --- SMOTE 적용 (불균형한 경우) ---
if y.value_counts()[0] / y.value_counts()[1] > 2 or y.value_counts()[1] / y.value_counts()[0] > 2:  # 2:1 이상 차이나면
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print("Class distribution after SMOTE:\n", pd.Series(y_resampled).value_counts(normalize=True))
else:
    print("No resampling needed.")
    X_resampled, y_resampled = X, y
    
    
# UMAP 모델 설정
umap_2d = cnv_plot.UMAP(n_components=2, random_state=42)
umap_3d = cnv_plot.UMAP(n_components=3, random_state=42)

# 차원 축소 수행
X_umap_2d = umap_2d.fit_transform(X_resampled)
X_umap_3d = umap_3d.fit_transform(X_resampled)

# --- 2D UMAP 시각화 ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_umap_2d[:, 0], y=X_umap_2d[:, 1], hue=y_resampled, palette='viridis', alpha=0.7)
plt.title("2D UMAP Visualization")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(title="Cancer")
plt.show()

# --- 3D UMAP 시각화 ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_umap_3d[:, 0], X_umap_3d[:, 1], X_umap_3d[:, 2], c=y_resampled, cmap='viridis', alpha=0.7)

ax.set_title("3D UMAP Visualization")
ax.set_xlabel("UMAP Dimension 1")
ax.set_ylabel("UMAP Dimension 2")
ax.set_zlabel("UMAP Dimension 3")
plt.colorbar(scatter, ax=ax, label="Cancer")
plt.show()
