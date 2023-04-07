import streamlit as st
import pandas as pd
from pycaret.classification import *

st.subheader('학습용 데이터 업로드')

# 학습용 데이터 업로드
st.title('데이터 업로드')
train_file = st.file_uploader('파일 업로드')

# 학습용 파일 업로드
if train_file:
    # pd.read_csv()
    train = pd.read_csv(train_file)

# 예측 컬럼
target = st.text_input('예측 컬럼명 입력')

train_btn = st.button('학습 시작')

if train_btn:
    # setup
    clf = setup(data=train, # 사용자가 업로드한 파일
            target=target, # 사용자가 입력한 예측 컬럼명
            session_id=123, 
            verbose=False, # False로 설정시 설정에 대한 출력생성을 생략합니다. (default=True)
           ) 
    
    # 학습 결과 출력 _ 알고리즘 성능 가장 좋은 3개만 담음
    best_model = compare_models(sort='Accuracy', n_select=3, fold=5)
    
    # 모델 블렌딩
    blended_models = blend_models(best_model, fold=5)
    
    # 학습 결과 받아오기
    result = pull()
    st.dataframe(result)
    
    # 모델 저장
    save_model(blended_models, 'classfication-01')

    
# CSV로 다운로드 
@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


# 예측용 데이터 업로드
st.subheader('예측용 데이터 업로드')
test_file = st.file_uploader('예측용 파일 업로드')

# 예측용 파일 업로드
if test_file:
    # 코드 입력
    test = pd.read_csv(test_file)
    
# 예측 버튼
test_btn = st.button('예측')

# 예측 버튼 클릭시
if test_btn:
    # 모델 로드 load_model()
    loaded_model = load_model('classfication-01')
    
    # 예측 predict_model()
    predictions = predict_model(estimator=loaded_model, data=test)
    
    # 결과 다운로드
    download_btn = st.download_button(
        "예측 결과 다운로드",
        convert_df(predictions),
        "prediction.csv",
        "text/csv",
        key='download-csv'
    )