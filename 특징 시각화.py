import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from io import StringIO
import requests

wide_data = pd.read_csv("wide_data.csv")
merged_data = pd.read_csv("merged_data.csv")
rf_pred_df = pd.read_csv("rf_predictions.csv")
rf_importance = pd.read_csv("rf_importance.csv")

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

# Feature Importance 시각화 (상위 20개)
plt.figure(figsize=(10, 6))
plt.barh(rf_importance['Feature'][:20], rf_importance['Importance'][:20])
plt.xlabel('Importance')
plt.ylabel('Chromosome Segment')
plt.title('Random Forest Feature Importance (Top 20)')
plt.gca().invert_yaxis()  # y축 순서 반전 (높은 중요도가 위로)
plt.show()

