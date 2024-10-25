import streamlit as st
import joblib
# 定义模型文件的完整路径
model_path = r'.\rf_model.pkl'

# 加载模型
loaded_model = joblib.load(model_path)
# 加载之前训练好的模型

# 创建Streamlit应用程序界面
st.title('Random Forest Model Deployment')
st.write('Enter some input features to make predictions:')

# 创建输入框用于用户输入特征
new_onset_shock = st.selectbox('new onset shock ', options=[0, 1])
DBIL = st.number_input('DBIL', min_value=0.00, max_value=50.00, step=0.01)
SOFA = st.number_input('SOFA', min_value=0, max_value=9, step=1)
MDR = st.selectbox('MDR', options=[0, 1])
los_of_ICU = st.number_input('Los_of_ICU', min_value=1, max_value=150, step=1)
TBIL = st.number_input('TBIL', min_value=1.0, max_value=100.0, step=0.1)
SCr = st.number_input('SCr', min_value=0.0, max_value=300.0, step=0.1)
pre_shock = st.selectbox('pre_shock', options=[0, 1])
area_of_burn = st.number_input('area_of_shock', min_value=20, max_value=100, step=1)
three = st.number_input('Ⅲ', min_value=0, max_value=80, step=1)
inhalation_damage = st.selectbox('inhalation_damage', options=[0, 1])
ALB = st.number_input('ALB', min_value=5.0, max_value=50.0, step=0.1)

# 定义应用程序行为
if st.button('Predict'):
    # 将用户输入的特征转换为模型所需的输入格式
    input_features = [[SOFA, new_onset_shock, MDR, los_of_ICU, TBIL, SCr, pre_shock, area_of_burn, three, inhalation_damage, ALB]]

    # 使用加载的模型进行预测
    prediction = loaded_model.predict(input_features)
    print(prediction)

    # 显示预测结果
    st.write(f'Prediction:{prediction[0]}')

# # 运行Streamlit应用程序
# if __name__ == '__main__':
#     st.run()

