import numpy as np
import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 定义模型文件的完整路径
model_path = 'model/rf_model.joblib'
scaler_path = 'model/rf_scaler.joblib'

# 加载模型
loaded_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
# 加载之前训练好的模型

# 创建Streamlit应用程序界面
st.title('Random Forest Model Deployment')
st.write('Enter some input features to make predictions:')

# 创建输入框用于用户输入特征
new_onset_shock = st.selectbox('new_onset_shock', options=["Yes", "No"])
DBIL = st.number_input('DBIL (μmol/L)', min_value=0.00, max_value=50.00, step=0.01)
SOFA = st.number_input('SOFA', min_value=0, max_value=9, step=1)
MDR = st.selectbox('MDR', options=["Yes", "No"])
los_of_ICU = st.number_input('Los_of_ICU (day)', min_value=1, max_value=150, step=1)
TBIL = st.number_input('TBIL (μmol/L)', min_value=1.0, max_value=100.0, step=0.1)
SCr = st.number_input('SCr (μmol/L)', min_value=0.0, max_value=300.0, step=0.1)
pre_shock = st.selectbox('pre_shock', options=["Yes", "No"])
area_of_burn = st.number_input('area_of_burn (%)', min_value=10, max_value=100, step=1)
three = st.number_input('Ⅲ (%)', min_value=0, max_value=100, step=1)
inhalation_damage = st.selectbox('inhalation_damage', options=["Yes", "No"])
ALB = st.number_input('ALB (g/L)', min_value=5.0, max_value=50.0, step=0.1)

if new_onset_shock == "Yes":
    new_onset_shock = 1
else:
    new_onset_shock = 0
if MDR == "Yes":
    MDR = 1
else:
    MDR = 0
if pre_shock == "Yes":
    pre_shock = 1
else:
    pre_shock = 0
if inhalation_damage == "Yes":
    inhalation_damage = 1
else:
    inhalation_damage = 0
if DBIL >= 7:
    DBIL = 1
else:
    DBIL = 0

# 定义应用程序行为
if st.button('Predict'):
    # 将用户输入的特征转换为模型所需的输入格式
    input_features = np.array([[area_of_burn, three, pre_shock, inhalation_damage, los_of_ICU, new_onset_shock, MDR, TBIL, ALB, SCr, SOFA]])
    df = pd.DataFrame(input_features, columns=["area_of_burn", "III", "pre_shock", "Inhalation_Damage", "LOS_of_ICU", "new_onset_shock", "MDR", "TBIL", "ALB", "SCr", "SOFA"])
    df = scaler.fit_transform(df)
    # 使用加载的模型进行预测
    prediction = loaded_model.predict_proba(df)
    print(prediction)
    f = round(float(prediction[0][1]), 3)

    # 显示预测结果
    st.write(f'Sepsis Probability Prediction:{f}')

# # 运行Streamlit应用程序
# if __name__ == '__main__':
#     # st.run()
#     loaded_model = joblib.load(model_path)
#     input_features = np.array(
#         [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
#     df = pd.DataFrame(input_features,
#                       columns=["area_of_burn", "III", "pre_shock", "Inhalation_Damage", "LOS_of_ICU", "new_onset_shock",
#                                "MDR", "TBIL", "ALB", "SCr", "SOFA"])
#     scaler = StandardScaler()
#     df = scaler.fit_transform(df)
#     # 使用加载的模型进行预测
#     prediction = loaded_model.predict_proba(df)
#     print(prediction)


