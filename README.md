# CNV-cancer_predict-

다음은 프로젝트에 대한 README 파일입니다. 프로젝트의 목적, 데이터 처리 과정, 사용된 모델, 실행 방법 등을 체계적으로 설명합니다.

---

# CNV 데이터를 활용한 암 예측 프로젝트 (CNV-cancer_predict)

## 1. 프로젝트 개요

본 프로젝트는 복제수 변이(Copy Number Variation, CNV) 데이터를 활용하여 암(Cancer) 발병 유무를 예측하는 머신러닝 모델을 개발하는 것을 목표로 합니다.

다양한 머신러닝 모델을 학습 및 평가하고, 최적의 모델을 선정합니다. 또한, 여러 모델의 예측 결과를 종합하는 앙상블 기법을 적용하여 예측 성능을 향상시키고자 합니다.

## 2. 프로젝트 구조 및 파일 설명

```
CNV-cancer_predict/
│
├── cnv_plot.py                 # UMAP을 이용한 데이터 시각화 코드
├── x 뽑기.py                   # 전처리된 데이터를 학습/테스트용으로 분리 및 저장하는 코드
├── 비교.py                     # 정상/암 데이터 간의 차이를 비교하고 추출하는 코드
├── 앙상블.py                   # 여러 모델의 예측 결과를 앙상블하는 코드
├── 전제+저장.py                # 데이터 전처리, 모델 학습, 평가 및 저장까지의 전체 파이프라인 코드
├── 특징 시각화.py              # 모델의 특징 중요도를 시각화하는 코드
├── 특징추출.py                 # 원본 데이터로부터 모델 학습에 사용할 특징을 추출하고 저장하는 코드
├── 파일_다운.py                # gdc-client를 이용하여 GDC 포털에서 데이터를 다운로드하는 코드
├── 파일만들기.py               # 다운로드된 원본 데이터를 CSV 파일로 병합하는 코드
├── README.md                   # 프로젝트 설명 파일
│
├── cnv_normal.csv              # (입력) 정상군 CNV 데이터
├── cnv_cancer.csv              # (입력) 암환자군 CNV 데이터
│
├── X_train.csv                 # (출력) 학습용 데이터 (Feature)
├── X_test.csv                  # (출력) 테스트용 데이터 (Feature)
├── y_train.csv                 # (출력) 학습용 데이터 (Label)
├── y_test.csv                  # (출력) 테스트용 데이터 (Label)
├── final_rf_importance.csv     # (출력) 최종 모델의 특징 중요도
├── *_best_model.pkl            # (출력) 각 머신러닝 모델의 최적 파라미터가 저장된 파일
└── ...
```

## 3. 실행 과정

### 3.1. 데이터 준비

1.  **데이터 다운로드 (`파일_다운.py`)**
    * `gdc-client`를 사용하여 GDC 데이터 포털에서 필요한 CNV 데이터를 다운로드합니다.
    * 다운로드할 파일 목록은 텍스트 파일(`cnv(normal_only).txt` 등) 형태로 관리됩니다.

2.  **데이터 병합 (`파일만들기.py`)**
    * 다운로드된 여러 개의 `tsv` 파일을 하나의 CSV 파일 (`normal.csv`, `cancer.csv`)로 병합합니다.
    * 이 과정에서 각 데이터에 해당하는 ID와 암 유무(정상: 0, 암: 1)를 표기합니다.

3.  **데이터 비교 및 차이 추출 (`비교.py`)**
    * 정상군 데이터와 암환자군 데이터 간의 차이가 있는 행을 추출하여 별도의 파일로 저장합니다.

### 3.2. 특징 공학 (Feature Engineering)

1.  **데이터 로드 및 병합 (`전제+저장.py`, `특징추출.py`)**
    * `cnv_normal.csv`와 `cnv_cancer.csv` 파일을 로드하여 하나의 데이터프레임으로 병합합니다.

2.  **염색체 구간화 (Chromosome Segmentation)**
    * 염색체(Chromosome)를 일정한 크기(예: 1Mb)의 구간(segment)으로 나눕니다.
    * 각 구간은 '염색체_시작위치_끝위치' 형태의 고유 ID를 갖게 됩니다.

3.  **가중 평균 계산**
    * 각 구간에 속하는 CNV 데이터의 'Segment_Mean' 값을 가중 평균하여 해당 구간의 대표값으로 사용합니다.
    * 가중치로는 'Segment_Length'(구간 길이)와 'Num_Probes'(측정된 프로브 개수)를 정규화하여 사용함으로써, 더 신뢰도 높은 데이터를 중요하게 반영합니다.

4.  **피벗 테이블 생성**
    * 환자(ID)를 행으로, 염색체 구간(Chromosome_segment)을 열로 하는 피벗 테이블을 생성합니다.
    * 이를 통해 각 환자별로 모든 염색체 구간의 CNV 값을 특징(feature)으로 갖는 데이터를 구성합니다.  아래는 정상군과 암환자군 데이터의 분포 차이를 시각화한 결과입니다.
    ![정상-암 데이터 분포 차이](image/DSMD.png)

### 3.3. 모델 학습 및 평가

1.  **데이터 불균형 처리 (SMOTE)**
    * 정상군과 암환자군 데이터의 비율 불균형 문제를 해결하기 위해 SMOTE(Synthetic Minority Over-sampling Technique)를 적용합니다. 이는 소수 클래스의 데이터를 합성하여 데이터 분포를 균일하게 만듭니다.

2.  **데이터 분할 (`x 뽑기.py`)**
    * 전처리 및 SMOTE가 적용된 데이터를 학습용(Train)과 테스트용(Test) 데이터로 분할합니다.
    * 이때 각 클래스의 비율이 유지되도록 `stratify` 옵션을 사용합니다.
    * 분할된 데이터는 `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` 파일로 저장됩니다.

3.  **모델 학습 및 하이퍼파라미터 튜닝 (`전제+저장.py`)**
    * 다음과 같은 다양한 머신러닝 모델을 사용하여 학습을 진행합니다.
        * Logistic Regression
        * Support Vector Machine (SVM)
        * Random Forest
        * Gradient Boosting
        * XGBoost
        * k-Nearest Neighbors (k-NN)
        * Gaussian Naive Bayes
    * `RandomizedSearchCV`를 사용하여 각 모델의 최적 하이퍼파라미터를 탐색합니다.
    * F1-score를 기준으로 최적 모델을 평가하고 선정합니다.

4.  **모델 평가 및 결과 저장**
    * 테스트 데이터에 대한 각 모델의 성능을 평가하고, Classification Report, Confusion Matrix, ROC-AUC, Precision-Recall Curve 등 다양한 지표를 통해 결과를 확인합니다.
    * 학습된 최적의 모델은 `joblib`을 사용하여 `.pkl` 파일 형태로 저장됩니다. 아래는 최종 모델의 ROC 커브와 Precision-Recall 커브입니다.
    ![ROC and PRC Curves](image/cnv_ROC_PRC.png)

### 3.4. 특징 분석 및 시각화

1.  **특징 중요도 분석 (`특징추출.py`, `특징 시각화.py`)**
    * Random Forest 모델의 `feature_importances_` 속성을 이용하여 암 예측에 중요한 영향을 미치는 염색체 구간(특징)을 식별합니다.
    * 중요도가 높은 상위 특징들을 막대그래프로 시각화하여 어떤 유전적 위치의 변이가 암과 관련이 깊은지 분석합니다.

    ![상위 20개 중요 변수](image/top20.png)

2.  **UMAP 시각화 (`cnv_plot.py`)**
    * 고차원 특징 공간을 2차원 또는 3차원으로 축소하여 시각화하는 UMAP(Uniform Manifold Approximation and Projection)을 사용합니다.
    * 이를 통해 정상군과 암환자군 데이터가 특징 공간에서 어떻게 분포하고 군집화되는지 직관적으로 확인할 수 있습니다.

### 3.5. 앙상블 모델링 (`앙상블.py`)

1.  **개별 모델 로드**
    * CNV 데이터뿐만 아니라, 메틸레이션(Methylation), 단백질(Protein) 등 다른 종류의 데이터(Multi-omics)로 학습된 개별 모델들을 불러옵니다.

2.  **가중 평균 앙상블**
    * 각 모델의 F1-score를 기반으로 가중치를 부여하여 예측 확률을 평균냅니다. 성능이 좋은 모델의 예측 결과에 더 높은 가중치를 주는 방식입니다.
    * 가중치 계산 시 제곱근을 취하여 특정 모델의 영향력이 과도해지는 것을 방지합니다. (`weights = {model: (f1 / total_f1) ** 0.5 ...}`)

3.  **최종 평가**
    * 앙상블 모델의 성능을 평가하고 ROC-AUC, Precision-Recall Curve 등을 통해 개별 모델과 성능을 비교 분석합니다.

## 4. 실행 방법

1.  `파일_다운.py`를 실행하여 GDC에서 원본 데이터를 다운로드합니다.
2.  `파일만들기.py`를 실행하여 다운로드된 데이터를 `cnv_normal.csv` 와 `cnv_cancer.csv` 파일로 통합합니다.
3.  `전제+저장.py`를 실행하여 데이터 전처리, 모델 학습 및 평가, 개별 모델 저장을 한 번에 수행합니다.
4.  (선택) `앙상블.py`를 실행하여 저장된 개별 모델들을 활용한 앙상블 모델링을 수행합니다.
5.  (선택) `cnv_plot.py` 및 `특징 시각화.py`를 실행하여 결과 데이터를 시각화합니다.