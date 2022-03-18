import streamlit as st
import pandas as pd
from datetime import datetime
from keras.models import load_model
import sys
import pandas as pd
import numpy as np
from numpy import array
import os
from sklearn.preprocessing import StandardScaler
import datetime
import tensorflow as tf
from tensorflow import keras
from keras import layers,callbacks,metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM,Bidirectional
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from matplotlib import rcParams
from sklearn.model_selection import GridSearchCV


def create_model(trainX,trainY,optimizer='adam'):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(trainX.shape[1],trainX.shape[2]),return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(25, activation='relu',return_sequences=False))
    model.add(Dense(trainY.shape[1]))
    model.compile(optimizer=optimizer, loss='mse')
    return model

def create_BD_LSTM_model(trainX,trainY,optimizer='adam'):
  modelBDNew = Sequential()
  modelBDNew.add(Bidirectional(LSTM(trainX.shape[2], activation='relu',return_sequences=True), input_shape=(trainX.shape[1],trainX.shape[2])))
  modelBDNew.add(Dropout(0.2))
  modelBDNew.add(Bidirectional(LSTM(trainX.shape[2], activation='relu',return_sequences=False), input_shape=(trainX.shape[1],trainX.shape[2])))
  modelBDNew.add(Dense(trainY.shape[1]))
  modelBDNew.compile(optimizer=optimizer, loss='mse')
  modelBDNew.summary()   
  
  return modelBDNew

def cleanTimeSeriesDF(df,cols):
    df.sort_values(by=['Day'])
    dateCols = ['year','Year','quarter',"Quarter","DayofYear"]
    newDF = df[cols]

    if 'Day' in cols:
      print("adding weekends/weekdays")
      conditions = [(pd.to_datetime(newDF['Day']).apply(lambda x: x.weekday()) >= 0) & (pd.to_datetime(newDF['Day']).apply(lambda x: x.weekday()) <= 4),(pd.to_datetime(newDF['Day']).apply(lambda x: x.weekday()) == 5)|(pd.to_datetime(newDF['Day']).apply(lambda x: x.weekday()) == 6)]
      values = ['weekday', 'weekend']
      newDF['week day/end'] = np.select(conditions, values)
      print("this worked")
    else:
      print("This did not work")
    
    for col in dateCols:
      if col in cols:
        newDF[col] = newDF[col].astype(object)
        print("this worked")
      else:
        print("this did not work either")  
    
    print(newDF.head())
    print(newDF.info())     
    

    raw_df = pd.get_dummies(newDF)
    print(raw_df.shape)
    print(raw_df.head())
    scaler = StandardScaler()
    scaler = scaler.fit(raw_df)
    raw_df_scaled = scaler.transform(raw_df)

    return raw_df_scaled,scaler,raw_df


def cleanModelData(df,cols):
    df.sort_values(by=['Day'])
    dateCols = ['year','Year','quarter',"Quarter","DayofYear"]
    newDF = df[cols]

    print(newDF.head())
    print(newDF.info())     
    

    raw_df = pd.get_dummies(newDF)
    print(raw_df.shape)
    print(raw_df.head())
    scaler = StandardScaler()
    scaler = scaler.fit(raw_df)
    raw_df_scaled = scaler.transform(raw_df)

    return raw_df_scaled,scaler,raw_df

def trinTestSplit(raw_df_scaled,train_dates):
    
    train_size = int(len(raw_df_scaled) * 0.90)
    test_size = len(raw_df_scaled) - train_size
    train, test = raw_df_scaled[0:train_size,:], raw_df_scaled[train_size:len(raw_df_scaled),:]
    print(len(train), len(test))
    trainDates, testDates = train_dates.values[0:train_size], train_dates.values[train_size:len(train_dates.values)]

    return train, test, trainDates, testDates

def trainTestWindowGenerator(train, test,window,n_future,cols,targetcol):    

    trainX = []
    trainY = []
    testX = []
    testY = []
    index = cols.index(targetcol[0])

    for i in range(window, len(train) - n_future+1):
        trainX.append(train[i - window:i,0:train.shape[1]])
        trainY.append(train[i + n_future - 1:i + n_future,index])
    trainX,trainY = np.array(trainX), np.array(trainY) 
    print(trainX.shape,trainY.shape)

    for i in range(window, len(test) - n_future+1):
        testX.append(test[i - window:i,0:test.shape[1]])
        testY.append(test[i + n_future - 1:i + n_future,index])
    testX,testY = array(testX), array(testY) 
    print(testX.shape,testY.shape)
    return trainX,trainY, testX,testY


def WindowGenerator(raw_df_scaled,window,n_future,cols,targetcol):    
    trainX = []
    trainY = []
    index = cols.index(targetcol[0])
    print(index)
    for i in range(window, len(raw_df_scaled) - n_future+1):
        trainX.append(raw_df_scaled[i - window:i,0:raw_df_scaled.shape[1]])
        trainY.append(raw_df_scaled[i + n_future - 1:i + n_future,index])
    trainX,trainY = array(trainX), array(trainY)  
    print(trainX.shape,trainY.shape)
    
    return trainX,trainY


n_future = 1
window = 30
#n_future2 = 30


@st.cache
def convert_df(df):
  # IMPORTANT: Cache the conversion to prevent computation on every rerun
  return df.to_csv().encode('utf-8')

def makeFuturePredictions(model,n_future,data,train_dates,scaler,originaldf,rawdf,predbatch,cols,targetcol,d,e):
  
  forecast_period_dates = pd.date_range(list(train_dates)[-e],periods=n_future,freq='1d').tolist()
  forecast_dates = []
  for time_i in forecast_period_dates:
    forecast_dates.append(time_i.date())
  targetcol = targetcol[0]
  forecast = model.predict(data[-n_future:],batch_size=predbatch)
  forecast_copies = np.repeat(forecast,rawdf.shape[1],axis=-1)
  y_pred_future = scaler.inverse_transform(forecast_copies)[:,0]

  df_forecast = pd.DataFrame({'Day':np.array(forecast_dates),'Predicted '+targetcol:y_pred_future})
  df_forecast['Day'] = pd.to_datetime(df_forecast['Day'])
  df_forecast['Predicted '+targetcol] = df_forecast['Predicted '+targetcol].shift(-1)
  df_forecast.dropna()
  original = originaldf[['Day',targetcol]]
  original.loc[:,'Day'] = pd.to_datetime(original.loc[:,'Day'])
  original = original.loc[original['Day'] >= "'"+str(d)+"'"]
  original.reset_index(inplace=True, drop=True)
  original.set_index('Day',inplace=True)
  df_forecast.set_index('Day',inplace=True)
  result = pd.concat([original,df_forecast],axis=1)
  
  st.line_chart(result)

  with st.expander("See explanation"):
    st.write("""
         The chart above shows some numbers I picked for you.
         I rolled actual dice for these, so they're *guaranteed* to
         be random.
    """)

  csv = convert_df(result)

  st.download_button(
    label="Download prediction data as CSV",
    data=csv,
    file_name='large_df.csv',
    mime='text/csv',
  )  


st.title("LSTM Nerual Net Visualisation")


import os


add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("Select a Model", "Create new Model","Conversions_Model_10_7_7_Adam","Conversions_Model_10_7_2_RMSprop","Conversions_Model_100_7_7_RMSprop","Conversions_Model_10_7_7_RMSprop","Conversions_Model_50_7_2_Adam")
)

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  df['year'] = df['year'].astype(object)
  columns = ["select all","Use Model"]+list(df.columns)
  columnSelection = st.sidebar.multiselect('Select columns to include in model', columns)
  targetcol = st.sidebar.multiselect('Select column you want to predict', list(df.columns))

  d = st.sidebar.date_input(
     "Select a start date  of the chart",
     datetime.date(2021, 7, 1))

  n_future2 = st.sidebar.slider('How many Days to predict', 0, 130, 25)
  
  backdate = st.sidebar.slider('How many Days to backdate prediction', 0, 130, 25)

  if("select all" in columnSelection):
    cols = list(df.columns)
    st.write(cols)
  elif "Use Model" in columnSelection:
    cols = ['Conversions','Clicks','Impr.','Cost','Conv. value','Store Visits','Store Revenue','dayName','Quarter2','year'] 
    st.write(cols)
  elif columnSelection:
    cols = []
    for x in columnSelection:
      cols.append(x)
    st.write(cols)  
  else:
    pass
  
  
  if (targetcol) and (add_selectbox == 'Create new Model'):
    train_dates = pd.to_datetime(df['Day'])
    raw_df_scaled,scaler,raw_df = cleanTimeSeriesDF(df,cols)
    train, test, trainDates, testDates = trinTestSplit(raw_df_scaled,train_dates)
    trainX,trainY, testX,testY = trainTestWindowGenerator(train, test,window,n_future,cols,targetcol)
    trainX2,trainY2 = WindowGenerator(raw_df_scaled,window,n_future,cols,targetcol)
  elif (targetcol) and ("model" in add_selectbox):
    train_dates = pd.to_datetime(df['Day'])
    raw_df_scaled,scaler,raw_df = cleanModelData(df,cols)
    train, test, trainDates, testDates = trinTestSplit(raw_df_scaled,train_dates)
    trainX,trainY, testX,testY = trainTestWindowGenerator(train, test,window,n_future,cols,targetcol)
    trainX2,trainY2 = WindowGenerator(raw_df_scaled,window,n_future,cols,targetcol)     
else:
  st.write('Please upload a data file')




if st.button('Click Me'):
  if add_selectbox == 'Create new Model':
    model = create_BD_LSTM_model(trainX,trainY,optimizer='Adam')
    early_stopping = callbacks.EarlyStopping(monitor='val_loss',patience=10)
    model.fit(trainX,trainY, epochs=100,validation_data=(testX,testY),verbose=1,batch_size=2,callbacks=[early_stopping])
    makeFuturePredictions(model,n_future2,trainX2,testDates,scaler,df,raw_df,7,cols,targetcol,d,backdate)
  elif (add_selectbox != "Select a Model" or add_selectbox != "Create new Model"):
    model = load_model(add_selectbox)
    train_dates = pd.to_datetime(df['Day'])
    cols = ['Conversions','Clicks','Impr.','Cost','Conv. value','Store Visits','Store Revenue','dayName','Quarter2','year'] 
    raw_df_scaled,scaler,raw_df = cleanModelData(df,cols)
    train, test, trainDates, testDates = trinTestSplit(raw_df_scaled,train_dates)
    trainX,trainY, testX,testY = trainTestWindowGenerator(train, test,window,n_future,cols,targetcol)
    trainX2,trainY2 = WindowGenerator(raw_df_scaled,window,n_future,cols,targetcol)
    makeFuturePredictions(model,n_future2,trainX2,testDates,scaler,df,raw_df,7,cols,targetcol,d,backdate)
  else:
    pass
else:
  st.write('Click button to get prediction')
