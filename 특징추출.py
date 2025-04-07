import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from io import StringIO
import requests

# 1. 데이터 로드 및 병합

# cnv_normal.csv 데이터 다운로드 및 DataFrame으로 읽기
url_normal = r"C:\Users\21\Desktop\cnv\cnv_cancer.csv"
normal_data = pd.read_csv(url_normal)
normal_data['Cancer'] = 0

# cnv_cancer.csv 데이터 다운로드 및 DataFrame으로 읽기
url_cancer = r"C:\Users\21\Desktop\cnv\cnv_normal.csv"
cancer_data = pd.read_csv(url_cancer)
cancer_data['Cancer'] = 1

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

# wide_data DataFrame을 CSV 파일로 저장
wide_data.to_csv("wide_data.csv", index=False)
print("저장완료!")


# --- X, y 생성 (wide_data 기반) ---
# X: wide_data에서 'ID', 'Cancer' 열을 제외한 모든 열
X = wide_data.drop(['ID','Cancer'], axis=1,errors = 'ignore')
X = X.fillna(X.mean()) # 결측치 처리

# y:  merged_data에서 ID기준으로 중복된 행 제거 후 Cancer 열
# (wide_data에 'Cancer' 열이 없을 수도 있으므로)

temp = merged_data.drop_duplicates(subset=['ID'])[['ID','Cancer']] # ID, Cancer만
y = pd.merge(wide_data,temp,on='ID',how='left')['Cancer'] # wide_data의 ID와 병합


# --- 데이터 분할 (stratify=y 적용) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nShape after train_test_split:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
# --- Feature Importance 계산 (예: Random Forest) ---

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

rf_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf.feature_importances_})
rf_importance = rf_importance.sort_values(by='Importance', ascending=False)
print("\nRandom Forest Feature Importance:\n", rf_importance)

# rf_importance DataFrame을 CSV 파일로 저장
rf_importance.to_csv("rf_importance.csv", index=False)

# --- Permutation Importance (옵션) ---

perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
perm_importance_df = pd.DataFrame({'Feature': X_test.columns, 'Importance': perm_importance.importances_mean})
perm_importance_df = perm_importance_df.sort_values(by='Importance', ascending=False)
print("\nPermutation Feature Importance:\n", perm_importance_df)

# --- merged_data 저장 ---
merged_data.to_csv("merged_data.csv", index=False)
print("\nmerged_data saved to merged_data.csv")

# --- Random Forest 모델로 예측한 결과 저장 ---
rf_predictions = rf.predict(X_test)  # 예측 결과
rf_pred_df = pd.DataFrame({'ID': wide_data.loc[X_test.index, 'ID'], 'Predicted_Cancer': rf_predictions}) # ID와 예측결과를 묶음
rf_pred_df.to_csv("rf_predictions.csv", index=False)
print("Random Forest predictions saved to rf_predictions.csv")