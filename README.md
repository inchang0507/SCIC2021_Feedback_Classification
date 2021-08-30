# scic2021_Feedback_Classification

1. 개요

1-1. 사전 학습 모델(RoBERTa)를 이용하여 발화의 종류를 분류

1-2. 프레임워크
- 전처리 >> 모델 학습 (RoBERTa-small, RoBERTa-large[smote 적용/미적용]) >> 앙상블(우선순위에 따라 선택)


2. 전처리 사항

2-1. 띄어쓰기(PyKoSpacing), 맞춤법 교정(py-hanspell)
- 제공 데이터에 적용하여 업로드 후 학습 데이터로 사용

2-2. 오버샘플링(SMOTE)
- 소수 클래스(2~50건) 데이터에 대해 SMOTE 오버 샘플링 기법을 적용하여 클래스 불균형 완화
- 텍스트 데이터를 TfidfVectorizer로 벡터화 시켜 SMOTE를 적용
- RoBERTa-large 모델 두 가지 중 하나에는 미적용



3. 모델 학습 및 앙상블

3-1. 3가지 모델을 활용, 제공된 1만건의 데이터를 학습 (27개 클래스)

3-2. 모델 앙상블하여 최종 결과 도출
- 3가지 모델 다수결 투표
- 결과가 모두 다른 경우, 제공된 우선순위에 따라 도출

※ 전처리 등의 기능을 담은 Class, def 등의 코드는 .py 파일로 저장 후 import 하여 사용
model 폴더 내 preprocess 폴더에 저장
Class_Roberta : RoBERTa용 데이터 세트 로딩
Preprocessor : 레이블 인코딩, 리스트 형태 변환, 테스트 데이터 전처리, 오버샘플링

※ 모델 및 패키지 출처 :
github.com/pytorch/fairseq/tree/master/examples/roberta
github.com/haven-jeon/PyKoSpacing
github.com/ssut/py-hanspell

※ 공모전 데이터인 관계로 데이터는 빼고 업로드, roberta_large 및 robert_small 모델은 용량상 제거 후 업로드