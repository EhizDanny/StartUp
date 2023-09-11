import streamlit as st
import pandas  as pd 
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import warnings
warnings.filterwarnings('ignore')
import joblib
from sklearn.linear_model import LinearRegression

data = pd.read_csv('startUpData.csv')

# load the model
model = joblib.load(open('LinearReg.pkl', 'rb'))

# ---------- Streamlit Development Starts ----------
st.markdown("<h1 style = 'text-align: center; color: #176B87'>START UP BUSINESS PREDICTOR</h1>", unsafe_allow_html = True)
st.markdown("<h6 style = 'text-align: center; top-margin: 0rem; color: #64CCC5'>BUILT BY Gomycode Yellow Orange Beast</h1>", unsafe_allow_html = True)

# st.title('START UP BUSINESS PREDICTOR')
# st.write('Built By Gomycode Yellow Orange Beast')
st.markdown("<br> <br>", unsafe_allow_html= True)

st.write('Pls Enter Your Username')
username = st.text_input('Please enter username: ')
if st.button('Submit Name'):
    st.success(f"Welcome {username}. Pls enjoy your usage")

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<h2 style = 'top-margin: 0rem;text-align: center; color: #AED2FF'>Project Introduction</h1>", unsafe_allow_html = True)

# Write project Introduction
st.markdown("<p style = 'text-align: justify; color: #AED2FF'>In the dynamic landscape of entrepreneurship, understanding and predicting the profitability of startup firms is crucial for investors, founders, and stakeholders alike. The project at hand employs a Linear Regression model to analyze and forecast the profitability of startup companies. Leveraging historical data on various startups, this study aims to unravel the key factors that influence a startup's success and provide valuable insights for making informed investment decisions. By delving into the intricate relationship between variables such as R&D expenditure, marketing spend, location, and industry, this project endeavors to shed light on the critical drivers of startup profitability, contributing to a more data-driven approach in the world of business.</p>", unsafe_allow_html = True)


# heat_map = plt.figure(figsize = (14,7)) # ----------------------------------------------------- Create a heat map plot
# correlation_data = data[['R&D Spend',	'Administration',	'Marketing Spend', 'Profit']] # --- Select data for correlation
# sns.heatmap(correlation_data.corr(), annot = True, cmap = 'BuPu')

# st.write(heat_map)
data.drop('Unnamed: 0', axis = 1, inplace = True)
st.write(data.sample(10))

st.sidebar.image('pngwing.com-2.png', width = 300, caption= f"Welcome {username}", use_column_width= True)

st.markdown("<br>", unsafe_allow_html= True)

# picture = st.camera_input('Take a picture')
# if picture:
#     st.sidebar.image(picture, use_column_width= True, caption = f"Welcome {username}")

st.sidebar.write('Pls decide your variable input type')
input_style = st.sidebar.selectbox('Pick Your Preferred input', ['Slider Input', 'Number Input'])

if input_style == 'Slider Input':
    research = st.sidebar.slider('R & D Spending', data['R&D Spend'].min(), data['R&D Spend'].max() )
    admin = st.sidebar.slider('Administration', data['Administration'].min(), data['Administration'].max())
    marketting = st.sidebar.slider('Marketting Expense', data['Marketing Spend'].min(), data['Marketing Spend'].max()) 
    states = data['State'].unique()
    state =  st.sidebar.select_slider('Select Your State Model', states)

else:
    research = st.sidebar.number_input('R & D Spending', data['R&D Spend'].min(), data['R&D Spend'].max() )
    admin = st.sidebar.number_input('Administration', data['Administration'].min(), data['Administration'].max())
    marketting = st.sidebar.number_input('Marketting Expense', data['Marketing Spend'].min(), data['Marketing Spend'].max()) 
    states = data['State'].unique()
    state =  st.sidebar.selectbox('Select Your State Model', states)

st.subheader("Your Inputted Data")
input_var = pd.DataFrame([{'R&D Spend': research, 'Administration': admin, 'Marketing Spend': marketting}])
st.write(input_var)

st.markdown("<br>", unsafe_allow_html= True)
tab1, tab2 = st.tabs(["Prediction Pane", "Intepretation Pane"])

with tab1:
    if st.button('PREDICT'):

        st.markdown("<br>", unsafe_allow_html= True)
        prediction = model.predict(input_var)
        st.write("Predicted Profit is :", prediction)
    else:
        st.write('Pls press the predict button for prediction')

with tab2:
    st.subheader('Model Interpretation')
    st.write(f"Profit = {model.intercept_.round(2)} + {model.coef_[0].round(2)} R&D Spend + {model.coef_[1].round(2)} Administration + {model.coef_[2].round(2)} Marketing Spend")

    st.markdown("<br>", unsafe_allow_html= True)

    st.markdown(f"- The expected Profit for a startup is {model.intercept_}")

    st.markdown(f"- For every additional 1 dollar spent on R&D Spend, the expected profit is expected to increase by ${model.coef_[0].round(2)}  ")

    st.markdown(f"- For every additional 1 dollar spent on Administration Expense, the expected profit is expected to decrease by ${model.coef_[1].round(2)}  ")

    st.markdown(f"- For every additional 1 dollar spent on Marketting Expense, the expected profit is expected to increase by ${model.coef_[2].round(2)}  ")