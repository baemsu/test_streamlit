
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import joblib
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import plotly.express as px


def user_input_features() :
  criteria_MNP = st.sidebar.number_input("MNP비율기준(예:0.45)")
  KT_10 = st.sidebar.number_input("10시 KT")
  total_10 =st.sidebar.number_input("10시 종합")
  KT_11 = st.sidebar.number_input("11시 KT")
  total_11 =st.sidebar.number_input("11시 종합")
  KT_12 = st.sidebar.number_input("12시 KT")
  total_12 =st.sidebar.number_input("12시 종합")
  KT_13 = st.sidebar.number_input("13시 KT")
  total_13 =st.sidebar.number_input("13시 종합")
  KT_14 = st.sidebar.number_input("14시 KT")
  total_14 =st.sidebar.number_input("14시 종합")
  KT_15 = st.sidebar.number_input("15시 KT")
  total_15 =st.sidebar.number_input("15시 종합")
  KT_16 = st.sidebar.number_input("16시 KT")
  total_16 =st.sidebar.number_input("16시 종합")
  KT_17 = st.sidebar.number_input("17시 KT")
  total_17 =st.sidebar.number_input("17시 종합")

# ['KT_10','KT_11','KT_12','KT_13','KT_14','KT_15','KT_16','total_10','total_16']]
  chart_16 ={"time" : ["10", "11", "12", "13", "14", "15", "16"],
             "KT_scale" : [KT_10,KT_11,KT_12,KT_13,KT_14,KT_15,KT_16],
             "total_scale" : [total_10, total_11,total_12,total_13,total_14,total_15,total_16]
             }

  chart_17 ={"time" : ["10", "11", "12", "13", "14", "15", "16","17"],
             "KT_scale" : [KT_10,KT_11,KT_12,KT_13,KT_14,KT_15,KT_16,KT_17],
             "total_scale" : [total_10, total_11,total_12,total_13,total_14,total_15,total_16,total_17]
             }


  data_KT_16 = {'KT_10' : KT_10,
          'KT_11' : KT_11,
          'KT_12' : KT_12,
          'KT_13' : KT_13,
          'KT_14' : KT_14,
          'KT_15' : KT_15,
          'KT_16' : KT_16,
          'total_10' : total_10,
          'total_16' : total_16,
          }
# [['KT_12','KT_13','KT_14','KT_15','KT_16','KT_17','total_12','total_17']]
  data_KT_17 = {'KT_12' : KT_12,
          'KT_13' : KT_13,
          'KT_14' : KT_14,
          'KT_15' : KT_15,
          'KT_16' : KT_16,
          'KT_17' : KT_17,
          'total_12' : total_12,
          'total_17' : total_17,
          }

# [['total_10','total_11','total_12','total_13','total_14','total_15','total_16','KT_12']]
  data_total_16 = {'total_10' : total_10,
          'total_11' : total_11,
          'total_12' : total_12,
          'total_13' : total_13,
          'total_14' : total_14,
          'total_15' : total_15,
          'total_16' : total_16,
          'KT_11' : KT_11,
          'KT_12' : KT_12,
          }
#['KT_12','total_12','total_13','total_14','total_15','total_16','total_17','KT_17']]
  data_total_17 = {'KT_12' : KT_12,
          'total_12' : total_12,
          'total_13' : total_13,
          'total_14' : total_14,
          'total_15' : total_15,
          'total_16' : total_16,
          'total_17' : total_17,          
          'KT_17' : KT_17,
          }

  data_chart_KT_16 = {"data":[KT_10,KT_11,KT_12,KT_13,KT_14,KT_15,KT_16]}

  features_KT_16 = pd.DataFrame(data_KT_16, index=[0])
  features_KT_17 = pd.DataFrame(data_KT_17, index=[0])
  features_total_16 = pd.DataFrame(data_total_16, index=[0])
  features_total_17 = pd.DataFrame(data_total_17, index=[0])
  features_chart_KT_16 = pd.DataFrame(data_chart_KT_16)

#  feature_16 = pd.DataFrame(chart_16, index=[0])
#  feature_17 = pd.DataFrame(chart_17, index=[0])


  return features_KT_16, features_KT_17, features_total_16, features_total_17, features_chart_KT_16, KT_16, total_16, criteria_MNP,KT_17, total_17, chart_16, chart_17



#def main():

st.header("<MVNO MNP 예측 솔루션>")

predict_time = st.radio("어떤 시간에 예측할 것인가요?",('오후4시', '오후5시'))

if predict_time == '오후4시':
  st.markdown('- **_오후 4시까지 데이터를 입력하고, 이를 가지고 예측해요._**')
  st.sidebar.header('User Input Parameters')
  df_KT_16, df_KT_17, df_total_16, df_total_17, df_chart_KT_16, KT_16, total_16, criteria_MNP, KT_17, total_17, chart_16, chart_17 = user_input_features()
  st.markdown("- **16시에 입력된 KT에서 KT로 온 수량**")
  st.table(df_KT_16)
  st.markdown("- **16시에 입력된 종합수량**")
  st.table(df_total_16)
  st.markdown("- **_16시 시점 MNP 비율_**")
  if total_16 == 0:
    st.write("입력중")
  else:
    st.write(KT_16/total_16)

  loaded_model = joblib.load("./regression_KT_16.pkl")
  predict_ = loaded_model.predict(df_KT_16)
  st.markdown("- **16시에 예측한 마감때의 KT 수량**")
  st.write(predict_[0])
  loaded_model2 = joblib.load("./regression_total_16.pkl")
  predict_2 = loaded_model2.predict(df_total_16)
  st.markdown("- **16시에 예측된 마감때의 종합 수량**")
  st.write(predict_2[0])
  st.markdown("- **_예상되는 KT에서 KT MNP비율_**")
  st.write(predict_[0]/predict_2[0])
#   st.line_chart(df_chart_KT_16)


#  st.line_chart(chart_16)

  chart_162 = pd.DataFrame(chart_16)
#  st.write(chart_162)

  chart_163 = chart_162.append({'time':'20' ,'KT_scale':float(predict_),'total_scale':float(predict_2)}, ignore_index=True)
  st.write(chart_163)

  chart_163["ratio"] = chart_163["KT_scale"]/chart_163["total_scale"]

  fig2 = px.bar(chart_163, x='time', y=['KT_scale','total_scale'])        #plotly bar차트
  st.plotly_chart(fig2)

  fig3 = px.bar(chart_163, x='time', y=['ratio'])        #plotly bar차트
  st.plotly_chart(fig3)

  st.markdown("- **최종 추가 가능한 수량 최대치**")

#  model_regression = joblib.load('./regression_KT_gap_16.pkl')
#  a= model_regression.coef_
#  b= model_regression.intercept_
  a=0.49906244085726026
  b=-14.99416837030202

  optim_no = int(predict_2) - int(total_16)

  for i in range(int(predict_2) - int(total_16)):
    MNP_percent_est = (KT_16 + a*(i+1) + b  )/(total_16 + (i+1))
    #print(i+1, round(MNP_percent_est,2))
    if(round(MNP_percent_est,2)>= float(criteria_MNP)):
      #print("45%를 넘지 않을 최대추가갯수", i)
      optim_no = i

  if total_16 == 0:
    st.write("입력중")
  else:
    if predict_/predict_2 > criteria_MNP and KT_16/total_16 > criteria_MNP:
      optim_no = "stop"    
      st.write(optim_no)
    else:
      st.write(optim_no)


else:
  st.markdown('- **오후 5시까지 데이터를 입력하고, 이를 가지고 예측해요.** ')
  st.sidebar.header('User Input Parameters')
#  df_KT_16, df_KT_17, df_total_16, df_total_17,df_chart_KT_16 = user_input_features()
  df_KT_16, df_KT_17, df_total_16, df_total_17, df_chart_KT_16, KT_16, total_16,criteria_MNP,KT_17, total_17,chart_16, chart_17 = user_input_features()
  st.markdown("- **17시에 입력된 KT에서 KT로 온 수량**")
  st.write(df_KT_17)
  st.markdown("- **17시에 입력된 종합수량**")
  st.write(df_total_17)
  st.markdown("- **17시 시점 MNP 비율**")
  if total_17 == 0:
    st.write("입력중")
  else:
    st.write(KT_17/total_17)

  loaded_model = joblib.load("./regression_KT_17.pkl")
  predict_ = loaded_model.predict(df_KT_17)
  st.markdown("- **17시에 예측한 마감때의 KT 수량**")
  st.write(predict_[0])
  loaded_model2 = joblib.load("./regression_total_17.pkl")
  predict_2 = loaded_model2.predict(df_total_17)
  st.markdown("- **17시에 예측된 마감때의 종합 수량**")
  st.write(predict_2[0])
  st.markdown("- **예상되는 KT에서 KT MNP비율**")
  st.write(predict_[0]/predict_2[0])

  chart_172 = pd.DataFrame(chart_17)
#  st.write(chart_172)

  chart_173 = chart_172.append({'time':'20' ,'KT_scale':float(predict_),'total_scale':float(predict_2)}, ignore_index=True)
  st.write(chart_173)

  chart_173["ratio"] = chart_173["KT_scale"]/chart_173["total_scale"]

  fig2 = px.bar(chart_173, x='time', y=['KT_scale','total_scale'])        #plotly bar차트
  st.plotly_chart(fig2)

  fig3 = px.bar(chart_173, x='time', y=['ratio'])        #plotly bar차트
  st.plotly_chart(fig3)

  st.markdown("- **최종 추가 가능한 수량 최대치**")
#  model_regression = joblib.load('./regression_KT_gap_17.pkl')
#  a= model_regression.coef_
#  b= model_regression.intercept_
  a=0.4935096103122253
  b=-6.953611748153236

  optim_no = int(predict_2) - int(total_17)

  for i in range(int(predict_2) - int(total_17)):
    MNP_percent_est = (KT_17 + a*(i+1) + b  )/(total_17 + (i+1))
    #print(i+1, round(MNP_percent_est,2))
    if(round(MNP_percent_est,2)>= float(criteria_MNP)):
      #print("45%를 넘지 않을 최대추가갯수", i)
      optim_no = i

  if total_17 == 0:
    st.write("입력중")
  else:
    if predict_/predict_2 > criteria_MNP and KT_17/total_17 > criteria_MNP:
      optim_no = "stop"    
      st.write(optim_no)
    else:
      st.write(optim_no)

#if __name__ == '__main__':
#	main()