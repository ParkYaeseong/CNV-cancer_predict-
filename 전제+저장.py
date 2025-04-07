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


# --- 데이터 분할 (stratify=y_resampled 적용) ---
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)
print("\nShape after train_test_split:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# --- Feature Importance (Random Forest) ---
# 초기 feature importance 계산 및 저장
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

rf_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf.feature_importances_})
rf_importance = rf_importance.sort_values(by='Importance', ascending=False)
print("\nInitial Random Forest Feature Importance:\n", rf_importance)

# 초기 feature importance 시각화 (상위 N개)
top_n = 20  # 상위 N개 feature
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=rf_importance.head(top_n), palette='viridis')
plt.title(f'Top {top_n} Initial Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig("initial_feature_importance.png")  # 이미지 저장
plt.show()


# --- 모델 정의 ---
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000), #max_iter 초기값
    'SVM': SVC(probability=True, random_state=42),  # probability=True for ROC
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),  # 수정
    'k-NN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

# --- 하이퍼파라미터 범위 정의 (RandomizedSearchCV용) ---
param_distributions = {
    'Logistic Regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
        'max_iter': [1000, 2000, 5000]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    },
    'Random Forest': {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'XGBoost': { # 수정
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'objective': ['binary:logistic'],  # 이진 분류
        'use_label_encoder': [False],
        'eval_metric': ['logloss']
    },
    'k-NN': {
        'n_neighbors': [3, 5, 7, 9],
        'weights' : ['uniform', 'distance'],
        'p' : [1,2]
    },
    'Naive Bayes': {}  # Naive Bayes는 일반적으로 튜닝할 하이퍼파라미터가 많지 않음
}
# --- 모델 학습, 튜닝, 평가, 저장 ---
results = {}
best_models = {}

for name, model in models.items():
    print(f"Training and tuning {name}...")
    try:
        if param_distributions[name]:
            random_search = RandomizedSearchCV(model, param_distributions[name],
                                                n_iter=10,
                                                cv=5,
                                                scoring='f1', #f1 스코어
                                                n_jobs = -1,
                                                random_state=42)
            random_search.fit(X_train, y_train)
            print(f" Best parameters for {name}: {random_search.best_params_}")
            print(f" Best cross-validation score (f1) for {name}: {random_search.best_score_:.4f}")
            best_model = random_search.best_estimator_
        else:
            model.fit(X_train, y_train)
            best_model = model

        #최종 모델 평가
        y_pred = best_model.predict(X_test)
        report = classification_report(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred)

        #F1 score 추가
        f1 = f1_score(y_test, y_pred)
        print(f"\n{name} - Final Evaluation:")
        print(report)
        print(matrix)
        print(f"F1 Score: {f1:.4f}")


        #결과 저장
        results[name] = {
            'report': report,
            'matrix': matrix,
            'best_params': random_search.best_params_ if param_distributions[name] else None,
            'best_score': random_search.best_score_ if param_distributions[name] else None,
            'f1_score': f1 #f1 스코어 저장
        }
        #모델 저장
        best_models[name] = best_model
        joblib.dump(best_model, f"{name}_best_model.pkl")

    except Exception as e:
        print(f"Error during training/tuning of {name}: {e}")
        results[name] = {'error': str(e)}

# 결과 출력 및 ROC, PR 곡선
print("\nResults Summary:")
results_df = pd.DataFrame.from_dict(results, orient='index')
print(results_df[['best_score', 'f1_score']]) #best_score, f1_score만 출력

plt.figure(figsize=(12, 6))

#ROC 곡선
plt.subplot(1, 2, 1) # 1행 2열 중 첫번째
for name, model in best_models.items():
    try:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else: #predict_proba 없는 경우 decision_function
            y_score = model.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    except Exception as e:
        print(f"Error plotting ROC for {name}: {e}")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")

#Precision-Recall 곡선
plt.subplot(1, 2, 2) # 1행 2열 중 두번째
for name, model in best_models.items():
    try:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:  # predict_proba가 없는 경우 decision_function 사용
            y_score = model.decision_function(X_test)

        precision, recall, _ = precision_recall_curve(y_test, y_score)
        average_precision = average_precision_score(y_test, y_score)
        plt.plot(recall, precision, label=f'{name} (AP = {average_precision:.2f})')

    except Exception as e:
        print(f"Error plotting PR curve for {name}: {e}")


plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()


# 가장 좋은 모델 선택 (F1-score 기준) - 이미 이전 코드에 포함됨
best_model_name = results_df['f1_score'].idxmax()
print(f"\nBest Model (based on F1-score): {best_model_name}")
best_model = best_models[best_model_name]  # 이미 딕셔너리에 저장되어 있음

# --- 최종 모델 Feature Importance (Best Model) ---
# best_model이 feature_importances_ 속성을 가지고 있는지 확인 후 사용

if hasattr(best_model, 'feature_importances_'):
    final_rf_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': best_model.feature_importances_})
    final_rf_importance = final_rf_importance.sort_values(by='Importance', ascending=False)
    print("\nFinal Feature Importance (Best Model):\n", final_rf_importance)

    # Feature Importance 시각화 (상위 N개, 예: 20개)
    top_n = 20
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=final_rf_importance.head(top_n), palette='viridis')
    plt.title(f'Top {top_n} Feature Importance ({best_model_name})')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f"final_feature_importance_{best_model_name}.png")  # 파일로 저장
    plt.show()

    # 중요 feature를 .csv파일로 저장
    final_rf_importance.to_csv("final_rf_importance.csv", index=False)

else:
  print(f"\n{best_model_name} does not have feature_importances_ attribute.")
  # 만약 coefficients를 볼 수 있는 모델이라면 (ex. LogisticRegression, LinearSVC)
  if hasattr(best_model, 'coef_'):
      print("\nModel Coefficients:")
      coef_df = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': best_model.coef_[0]})
      coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
      print(coef_df)
      # 시각화, 저장 (coef_df 사용) ...
