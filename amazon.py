import streamlit as st
from matplotlib.backends.backend_agg import RendererAgg
import pandas as pd
import seaborn as sns 
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib
import requests
from streamlit_lottie import st_lottie
from statsmodels.tsa.arima.model import ARIMA
import numpy as np


st.set_page_config(layout="wide")
sns.set_style('darkgrid')

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_amazon = load_lottieurl('https://assets1.lottiefiles.com/packages/lf20_WaPNTK.json')
st_lottie(lottie_amazon, speed=1, height=200, key="initial")


matplotlib.use("agg")
matplotlib.rcParams.update({'font.size': 14})

_lock = RendererAgg.lock

sns.set_style('darkgrid')
row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.beta_columns(
    (.1, 2, .2, 1, .1))

row0_1.title('Analyzing Your Amazon Purchase Habits')

with row0_2:
    st.write('')

row0_2.subheader(
    'A Web App by [Tyler Richards](http://www.tylerjrichards.com)')

row1_spacer1, row1_1, row1_spacer2 = st.beta_columns((.1, 3.2, .1))

with row1_1:
	'''
	Hey there! Welcome to my Amazon Analysis App. I found that I was spending an 
	insane amount on Amazon every year, but I didn't have a great idea about what
	exactly I was spending it on. I tried a few methods, like [different budgeting apps](https://copilot.money/)
	and [Amazon's puchase history page](https://www.amazon.com/gp/your-account/order-history)
	, but none of them gave me enough detail.    
	This app analyzes your Amazon purchase habits over time, the items you've bought, 
	how much you've spent, and where this is trending over time. Have fun!
	**To begin, please download [Amazon order history](https://www.amazon.com/gp/b2b/reports). 
	This app neither records nor stores your data, ever.**
	'''
	file_name = st.file_uploader('Upload Your Amazon Data Here')

if file_name is not None:
	df = pd.read_csv(file_name)
else:
	st.stop()

#data cleaning
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Item Total'] = df['Item Total'].replace({'\$':''}, regex=True)
df['Item Total'] = pd.to_numeric(df['Item Total'])
df['Order Year'] = df['Order Date'].dt.year
df['Order Month'] = df['Order Date'].dt.strftime('%B')
df['Order Month Digit'] = df['Order Date'].dt.month

#spend per year
df_orders_year = pd.DataFrame(df.groupby('Order Year').sum()['Item Total']).reset_index()

fig_spend_per_year = Figure(figsize=(8,7), dpi=900)
ax_spend_per_year = fig_spend_per_year.subplots()
sns.barplot(data=df_orders_year, 
	x='Order Year', y='Item Total', 
	palette='viridis', ax=ax_spend_per_year)
ax_spend_per_year.set_ylabel('Amount Spent ($)')
ax_spend_per_year.set_xlabel('Date')
ax_spend_per_year.set_title('Amazon Purchase Total By Year')
max_val = df_orders_year['Item Total'].max()
max_year = list(df_orders_year[df_orders_year['Item Total'] == max_val]['Order Year'])[0]

#st.pyplot(fig_spend_per_year)


#orders over time
df_copy = df.copy()
df_copy.set_index('Order Date', inplace=True)
df_month_date = pd.DataFrame(df_copy.resample('1M').count()['Order ID']).reset_index()
df_month_date.columns = ['date', 'count']

fig_orders_over_time = Figure(figsize=(8,7), dpi=900)
ax_orders_over_time = fig_orders_over_time.subplots()
sns.lineplot(data = df_month_date, x='date', 
	y='count', palette='viridis', ax=ax_orders_over_time)
ax_orders_over_time.set_ylabel('Purchase Count')
ax_orders_over_time.set_xlabel('Date')
ax_orders_over_time.set_title('Amazon Purchases Over Time')


#orders over month
df_month = df.groupby(['Order Month', 'Order Month Digit']).count()['Order Date'].reset_index()
df_month.columns = ['Month', 'Month_digit', 'Order_count']
df_month.sort_values(by='Month_digit', inplace=True)
fig_month = Figure(figsize=(8,7), dpi=900)
ax_month = fig_month.subplots()
sns.barplot(data = df_month, palette='viridis', 
	x = 'Month', y = 'Order_count', ax=ax_month)
ax_month.set_xticklabels(df_month['Month'], rotation=45)
ax_month.set_title('Amazon Shopping: Month Trend')
ax_month.set_ylabel('Purchase Count')
max_month_val = list(df_month.sort_values(by='Order_count', ascending=False).head(1)['Order_count'])[0]
max_month = list(df_month[df_month['Order_count'] == max_month_val]['Month'])[0]

#orders per city
df_cities = pd.DataFrame(df['Shipping Address City'].str.upper().value_counts()).reset_index()
df_cities.columns = ['City', 'Order Count']
fig_cities = Figure(figsize=(8,7), dpi=900)
ax_cities = fig_cities.subplots()
sns.barplot(data = df_cities, palette='viridis', x='City', y='Order Count', ax=ax_cities)
ax_cities.set_xticklabels(df_cities['City'], rotation=45)
ax_cities.set_title('Where Have Your Amazon Packages Gone?')


#order categories
df_cat = df.groupby(['Category']).count()['Order Date'].reset_index()
df_cat.columns = ['Category', 'Purchase Count']
df_cat.sort_values(by='Purchase Count', ascending=False, inplace=True)
df_cat = df_cat.head(15)
fig_cat = Figure(figsize=(8,7), dpi=900)
ax_cat = fig_cat.subplots()
sns.barplot(data = df_cat, palette='viridis', 
	x = 'Category', y = 'Purchase Count', ax=ax_cat)
ax_cat.set_xticklabels(df_cat['Category'], rotation=45, fontsize=8)
ax_cat.set_title('Top 15 Purchase Categories')
ax_cat.set_ylabel('Purchase Count')
pop_cat = list(df_cat.head(1)['Category'])[0]

#month prediction, moving average
data = list(df_month_date['count'])
# fit model
model = ARIMA(data, order=(0, 0, 1))
model_fit = model.fit()
# make prediction
yhat = np.round(model_fit.predict(len(data), len(data))[0])

st.write('## **Amazon Purchasing Over Time**')
st.write('-------------------')
col1, col2 = st.beta_columns(2)

with col1:
	st.pyplot(fig_spend_per_year)
	st.write('This graph showed me that I have depended more and more on Amazon for commerce over time, and especially when there were too many COVID cases in the US to go shopping. Looks like your biggest spending year was {} when you spent ${} on Amazon.'.format(max_year, round(max_val)))

with col2:
	st.pyplot(fig_orders_over_time)
	st.write('For me, this graph was useful because I could see two big upticks, once when I graduated high school and moved in for college and the second when I got my first internships and could actually buy more. I also made a simple moving average model on your data, and predict that next month you will buy {} items.'.format(yhat))

st.write('## **More Item Specific Analysis**')
st.write('-------------------')
col3, col4, col5 = st.beta_columns(3)

with col3:
	st.pyplot(fig_month)
	st.write('Over time, you have bought the most items in {}, a total of {} items. My biggest Amazon month was January, but only because I moved during two Januaries!'.format(max_month, max_month_val))

with col4:
	st.pyplot(fig_cities)
	st.write('I love this graph because it so clearly showed me where I have moved over time!')

#col5, col6 = st.beta_columns(2)
with col5:
	st.pyplot(fig_cat)
	st.write("My biggest category here by far was books, I've bought 3x more books than any other category! Your most popular category was {}".format(pop_cat))

st.write('## **RocketShip**')
st.write('-------------------')