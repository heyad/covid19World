
from datetime import date
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from fbprophet import Prophet
import pycountry
import plotly.express as px
import plotly.io as pio
from datetime import timedelta
from functools import reduce 
from plotly.subplots import make_subplots
from sklearn.impute import SimpleImputer

def read_files(date_update = date.today().isoformat()):
    # time seriese data 
    url1 = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    url2 = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
    url3 = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"
    df_confirmed = pd.read_csv(url1)
    df_deaths = pd.read_csv(url2)
    df_recovered = pd.read_csv(url3)

    # rename some columns 
    df_confirmed.rename(columns={'Country/Region':'Country'}, inplace=True)
    df_recovered.rename(columns={'Country/Region':'Country'}, inplace=True)
    df_deaths.rename(columns={'Country/Region':'Country'}, inplace=True)

    # get the dates 
    df_confirmed=df_confirmed.reset_index(drop=True)
    
    df = pd.read_csv('countries.csv')
    df.columns = ['Index','Country']
    #st.write(df.head(10))

    countries = df['Country'].to_list()
    
    return (df_confirmed,df_deaths,df_recovered,countries)

# Read data from John Hopkings 



##st.write(df_confirmed.head())
#                                               Prepare the data                #####
#                                               reshape the data frames         #####
#####################################################################################
def prepare_data(date_update = date.today().isoformat()):
    cols_list = df_confirmed.columns.to_list()[:4]
    dates_list = df_confirmed.columns.to_list()[4:]
    # tidy data df_confirmed 
    df_confirmedM = pd.melt(df_confirmed, id_vars=cols_list,\
         value_vars=dates_list, var_name='Date', value_name='Confirmed')

    # Deaths series 
    cols_list = df_deaths.columns.to_list()[:4]
    dates_list = df_deaths.columns.to_list()[4:]
    # tidy data df_deaths 
    df_deathsM = pd.melt(df_deaths, id_vars=cols_list,\
         value_vars=dates_list, var_name='Date', value_name='Deaths')

    # Recovered 
    cols_list = df_recovered.columns.to_list()[:4]
    dates_list = df_recovered.columns.to_list()[4:]

    #dates_list
    # and finally tidy data df_recovered 
    df_recoveredM = pd.melt(df_recovered, id_vars=cols_list,\
         value_vars=dates_list, var_name='Date', value_name='Recovered')
    # Merege the three time series into one
    df_all = [df_confirmedM, df_deathsM,df_recoveredM]          
    covid19 = reduce(lambda left, right: pd.merge(left, right, on =cols_list+['Date'], how='outer'), df_all)

    # Rename Palestine 
    covid19.loc[covid19.Country=='West Bank and Gaza','Country']='Palestine'
    # Tidy the df again: Rows to be represented by state, country, lat, long, and date 
    df_covid19 = covid19.copy()
    cols_ids = df_covid19.columns[:5]

    cases = ['Confirmed', 'Deaths','Recovered']
    df_covid19 = pd.melt(df_covid19, id_vars=cols_ids,\
              value_vars=cases, var_name='Cases', value_name='Count')
    df_covid19['Date'] = pd.to_datetime(df_covid19['Date'],format='%m/%d/%y', errors='raise')
    df_covid19['Week']=df_covid19['Date'].dt.strftime('%W')

    covid19['Active']=covid19['Confirmed']-covid19['Deaths']
    covid19['Date']=pd.to_datetime(covid19['Date'])
    df_grouped = covid19.groupby(['Country', 'Date'], as_index=False).agg({'Confirmed':'sum','Deaths':'sum',
                                                            'Active':'sum','Recovered':'sum'})
    df_grouped = df_grouped.groupby('Country')['Confirmed', 'Deaths','Recovered', 'Active'].max().reset_index()
    df_grouped = df_grouped.sort_values(by='Confirmed', ascending=False)
    df_grouped = df_grouped.reset_index(drop=True)
    # top 10 names 
    top_confirmed = df_grouped['Country'].to_list()
    # missing values (simple approach)
    df_covid19.fillna(0,inplace=True)
    covid19.fillna(0,inplace=True)
    df_grouped.fillna(0,inplace=True)
    # called again at 
    #st.write('called again at: '+ str(date_update))
    return(covid19,df_covid19,df_grouped)
