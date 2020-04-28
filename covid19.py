#!/usr/bin/env python
# coding: utf-8


# essential libraries
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


## import util functions 
import util_functions as util_f
# streamlit 
import streamlit as st


# #### Loading & Pre-processing files from github

st.title('Covid19')
#stdate =pd.to_datetime(max(covid19['Date']))

read_and_cache_csv = st.cache(pd.read_csv)


@st.cache  #  This function will be cached
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
@st.cache
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


df_confirmed, df_deaths,df_recovered,top_conf= read_files(date.today().isoformat())
covid19,df_covid19,df_grouped = prepare_data(date.today().isoformat())


### Fill missing values with zeros 

# EU countries 
eus = ['Spain','Belgium','Bulgaria','Croatia','Cyprus','Czech Republic','Denmark',
             'Estonia','Finland','France','Germany','Greece','Hungary',
            'Ireland','Italy','Latvia','Lithuania','Luxembourg','Malta',
             'Netherlands','Poland','Portugal','Romania' ,'Slovakia','Slovenia','Sweden','United Kingdom']
# Arab World 
arabs = ['Jordan','Egypt','Saudi Arabia','Qatar','Bahrain','Iraq','Algeria','Morocco','Lebanon','Syria',\
            'Kuwait','United Arab Emirates','Yemen','Sudan','Oman','Tunisia','Mauritania',\
         'Lybia','Union of the Comoros','Somali','Palestine']


# by deaths
df_grouped.sort_values(by='Deaths',ascending=False,inplace=True)
top_death = df_grouped['Country'].to_list()
# by recovered
df_grouped.sort_values(by='Recovered',ascending=False,inplace=True)
top_rec = df_grouped['Country'].to_list()

# Top 20 European Countries
df_eus = df_grouped[df_grouped.Country.isin(eus)].reset_index()
df_eus.columns=['World Rank','Country','Confirmed','Deaths','Recovered','Active']
df_eus['World Rank']=df_eus['World Rank']+1

x = df_eus.groupby('Country')['Confirmed', 'Deaths','Recovered', 'Active'].max().reset_index()
x.sort_values('Confirmed',ascending=False,inplace=True)
top_conf_eus = x['Country'].to_list()

x.sort_values('Deaths',ascending=False,inplace=True)
top_death_eus = x['Country'].to_list()

x.sort_values('Recovered',ascending=False,inplace=True)
top_rec_eus = x['Country'].to_list()

df_arabs = df_grouped[df_grouped.Country.isin(arabs)].reset_index()
df_arabs.sort_values('Recovered',ascending=False,inplace=True)
top_rec_arabs = df_arabs['Country'].to_list()
df_arabs.sort_values('Confirmed',ascending=False,inplace=True)
top_conf_arabs = df_arabs['Country'].to_list()

df_arabs.sort_values('Deaths',ascending=False,inplace=True)
top_death_arabs = df_arabs['Country'].to_list()

latest_no = pd.DataFrame({'Index':[1],'Confirmed':df_grouped['Confirmed'].sum(),
                          'Deaths':df_grouped['Deaths'].sum(),
                          'Recovered':df_grouped['Recovered'].sum(),
                          'Active': df_grouped['Active'].sum()})

def top_countries_by_cases_by_date(top=30,least=False,
                                   byDate='28.01.2020', cases='Confirmed',title='28.03.2020'):
    
    # The code below should generate similar barplot to the one generated above using df

    temp = covid19.copy()
    temp['Date']=pd.to_datetime(temp['Date'])

    mask = (temp['Date'] <= byDate)
    temp = temp.loc[mask]

    temp = temp.groupby(['Country', 'Date'], as_index=False).agg({'Confirmed':'sum','Deaths':'sum',
                                                             'Active':'sum','Recovered':'sum'})

    temp = temp.groupby('Country')['Confirmed', 'Deaths','Recovered', 'Active'].max().reset_index()
    temp = temp.sort_values(by=cases, ascending=False)
    temp = temp.reset_index(drop=True)
    
    if least==True:
        temp = temp[:top]
    else:
        x = temp.shape[0]
        x = x - top
        temp = temp[x:]
    
    if cases=='Confirmed':
        colors = 'rgb(26, 30, 250)'
    elif cases=='Deaths':
        colors = 'rgb(255, 60, 30)'
    else:
        colors = 'rgb(100, 255, 150)'
    #colors = ['deepskyblue',] * 5
    #colors[3] = 'crimson'

    fig = go.Figure(data=[go.Bar(
        x=temp['Country'],
        y=temp[cases],
        text=temp[cases],
        marker_color=colors
        #marker_color=colors # marker color can be a single color value or an iterable
    )])
    #byDate.strftime("%A %d. %B %Y")
    #byDate.strftime("%d/%m/%y")
    #fig.update_layout(showlegend=False)

    fig.update_layout(template='plotly_white')
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_yaxes(title_text="Number of Cases", hoverformat=".3f")
    fig.update_layout(title_text="Number of " + cases + " Cases: Top "  + str(top) +' Countries ' +title, title_x=0.5)
    return (fig)
def place_value(number): 
    return ("{:,}".format(number)) 
  

total_conf = df_grouped['Confirmed'].sum()
total_conf = int(total_conf)
total_deaths = df_grouped['Deaths'].sum()
total_deaths =int(total_deaths)

total_rec = df_grouped['Recovered'].sum()
total_rec =int(total_rec)



number_s = 5

number_s = st.sidebar.number_input('Top Countries',1,df_covid19.shape[0],5)

#### Start date end date x
start_date_df = pd.to_datetime(min(df_covid19['Date']))
end_date_df = pd.to_datetime(max(df_covid19['Date']))

start_date = st.sidebar.date_input('Start Date',start_date_df)
end_date = st.sidebar.date_input('End Date',end_date_df)




# top_w = st.sidebar.checkbox('Top Countries by confirmed cases worldwide',0)

# top_e = st.sidebar.checkbox('Top Countries EU',0)
# top_arabs = st.sidebar.checkbox('Top Countries(Arab World)',0)



#st.sidebar.write("List top countries by number of confirmed cases")
regions = st.sidebar.radio("Top countries by confirmed cases", 
    ['Worldwide','Europe','Arab World'])

covid19_cases = st.sidebar.selectbox('Select Cases',['Confirmed','Deaths','Recovered'])

plot_type = st.sidebar.radio('Chose plot type',['Bar','line'])
log_plot = st.sidebar.checkbox('logarithmic? ',0)
plot_bar = (plot_type=='Bar')
plot_line = (plot_type=='Line')

def show_top_countries_list(n=10):
    if regions=='Worldwide':
        #st.subheader('World')
        df_grouped.sort_values(by='Confirmed',ascending=False,inplace=True)
        st.markdown('<small>Top <strong>' + str(number_s) + '</strong> countries (worldwide) by number of confirmed covid19 cases</small>',True)
        st.write(df_grouped[:n].style.background_gradient(cmap='Reds'))
    if regions=='Europe': 
        #st.subheader('Europe')
        df_eus.sort_values(by='Confirmed',ascending=False,inplace=True)
        st.markdown('<small>Top <strong>' + str(number_s) + '</strong> countries (Europe) by number of confirmed covid19 cases</small>',True)
        st.write(df_eus[:n].style.background_gradient(cmap='Oranges'))
        #st.table(df_eus[:n])
    if regions=='Arab World':
        #st.subheader('Arab World')
        df_arabs = df_grouped[df_grouped.Country.isin(arabs)].reset_index()
        df_arabs.columns=['World Rank','Country','Confirmed','Deaths','Recovered','Active']
        df_arabs['World Rank']=df_arabs['World Rank']
        st.markdown('<small>Top <strong>' + str(number_s) + '</strong> countries (Arab World) by number of confirmed covid19 cases</small>',True)
        st.write(df_arabs.head(n).style.background_gradient(cmap='Reds'))

### By days the spread of covid19 ##### 
max_n_days =pd.to_datetime(max(covid19['Date']))-pd.to_datetime(min(covid19['Date']))
days_dif = max_n_days.days
st.markdown('In '+'<b>'+str(days_dif) + '</b> Days of Covid19, ' +'total number of confirmed cases worldwide  <b> ' +
    '</b> is <b>'+ 
    place_value(total_conf) +'</b>, the number of deaths is <b>'+
    str(place_value(total_deaths))+'</b>, and the total number of recovered cases is <b>'+
    place_value(total_rec)+'</b>. ' + 'Data is extracted from John Hopkins University [Github Repository](https://github.com/CSSEGISandData/COVID-19)'+
    ' and last updated on <b>'+str(pd.to_datetime(max(covid19['Date'])).strftime("%d-%m-%y"))+'</b>' +'. <p>Code is availble at [https://github.com/heyad/covid19World](https://github.com/heyad/covid19World)',True)
covid_days = 1


top_on_bar = st.checkbox('Plot top Countries by Cases',1)


def plot_countries_by_cases(start_date,end_date):
    if top_on_bar:
        
        number = 30
        
        fig = top_countries_by_cases_by_date(number,True,end_date,covid19_cases,
            ' by '+str(end_date.strftime("%d/%m/%y")))
        #streamlit 
        st.plotly_chart(fig) 

#def plot_countries_daily_s():

plot_countries_by_cases(start_date,end_date)


#### End of plotting section for top countries (barplots)
daily_spread = st.sidebar.checkbox('Daily Spread (Top countries)',1)


n = number_s 
# pass list of counries, (function is not accurate, specially when more than one country)
def plot_countries_daily(countries='all',cases='Confirmed',startDate='2020-1-21',
                         endDate='2020-3-22',title='Date',facet_cols=2,bar=False,line=True,logs=False):
    
    temp = covid19.loc[covid19['Country'].isin(countries),:].copy()
    #temp = covid19[(covid19.Country.isin(countries))].copy()
    temp['Date'] = pd.to_datetime(temp['Date'])

    #temp['Daily'] = df.groupby(['Country', 'Date'])[cases].diff().fillna(0)

    temp = temp.groupby(['Date', 'Country'])[cases].sum().reset_index().sort_values(cases, ascending=True)
    #temp['Daily'] = temp[cases].diff()
    #temp['Daily']=0
    
    mask = (temp['Date'] >= pd.to_datetime(startDate))&(temp['Date'] <= pd.to_datetime(endDate))
    temp = temp.loc[mask].copy()
    temp.sort_values(by=['Country', 'Date'])

    temp['Daily'] = temp.groupby('Country')[cases].diff()

    # start from first case
    temp = temp[temp.Daily>0].copy()
    temp['Mean'] = temp.iloc[:,3].rolling(window=7).mean()
    #st.write(temp.head())
    #temp['FirstCase'] = 0 
    #mask1 = temp['Daily']>=1 
    #temp = temp.loc[mask1].copy()
    
    #if startDate < pd.to_datetime(min(temp.loc['Date'])):
     #   startDate = pd.to_datetime(min(temp.loc['Date']))
    #if temp['Daily']>=1:

    #firstCase = pd.to_datetime(min(temp.lo['Date']))
    #st.write('final function call ',startDate,endDate)
    temp.fillna(0,inplace=True)
    #st.write(temp['Daily'])
    
    # impute with mean values to show overall graph trend 
    #temp['Daily'] = temp['Daily'].fillna((temp['Daily'].mean()))
    #temp['Date'] = temp['Date'].dt.strftime('%d-%m-%Y')

    if (bar==False):
        fig = px.line(temp, x="Date", y='Daily', color='Country',  height=500,
           facet_col='Country', facet_col_wrap=facet_cols,template='plotly_white')
    else:
        fig = px.bar(temp, x="Date", y='Daily', color='Country',  height=400,
           facet_col='Country', facet_col_wrap=facet_cols,template='plotly_white',text=temp['Daily'])
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

    #fig.update_yaxes(type="log")
    if (logs==True):
        fig.update_yaxes(type="log")
    #st.write(temp)
        
    title = title+'('+regions+') '+'['+cases+'] '+ 'between ' +str(startDate) + ' and ' + str(endDate)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_xaxes(tickangle=90)

    #temp['Date'] = temp['Date'].dt.strftime('%d-%m-%Y')

    #fig.update_layout(xaxis={'tickformat':"%b %d %Y "
     #           ,'type':'category'
    #})
    fig.update_layout(xaxis_title="Date",yaxis_title="Daily Count")
    fig.update_layout(title_text=title, title_x=0.5,width=900,height=550)
    return(fig)


def plot_countries_daily_s(startDate,endDate,n):
     
        
        n = 10
        if regions=='Worldwide':
            st.plotly_chart(plot_countries_daily(top_conf[:number_s],
                covid19_cases,startDate,endDate,
                             ' Covid19 daily/ ',4,plot_bar,plot_line))
        if regions =='Europe':
             st.plotly_chart(plot_countries_daily(top_conf_eus[:number_s],
                covid19_cases,startDate,endDate,
                             ' Covid19 daily/ ',4,plot_bar,plot_line))
        if regions =='Arab World':
             st.plotly_chart(plot_countries_daily(top_conf_arabs[:number_s],
                covid19_cases,startDate,endDate,
                             ' Covid19 daily/ ',4,plot_bar,plot_line))
#### Countries totoal 
def plot_cases_countries_totals_date(countries="all",cases='Confirmed',startDate="2020-3-1",
                                     endDate="2020-3-1",bars=False,
                                ncols=3,logs=False):
        temp = covid19[covid19.Country.isin(countries)]
        mask = (temp['Date'] >= pd.to_datetime(startDate))&(temp['Date'] <= pd.to_datetime(endDate))
        
        temp = temp.loc[mask]

        temp = temp.groupby(['Date', 'Country'])[cases].sum().reset_index().sort_values(cases,ascending=True)
        if (bars==True):
            fig = px.bar(temp, x='Date', y=cases, color='Country',text=cases,height=400,\
             template='plotly_white',facet_col='Country',facet_col_wrap=ncols) 
            fig.update_traces(texttemplate='%{text:.2s}', textposition='auto')

        elif logs==True: 
            fig = px.line(temp, x='Date', y=cases, color='Country',height=400,\
             template='plotly_white',facet_col='Country',facet_col_wrap=ncols)
            fig.update_yaxes(type="log")
        else:
            fig = px.line(temp, x='Date', y=cases, color='Country',\
             template='plotly_white',facet_col='Country',facet_col_wrap=ncols)       
            
        fig.update_layout(title_text=cases+ ' Covid19 Cases (cumulative) for top 10 countries (now)' + 
                           ' between '+str(startDate) +' and '+str(endDate),title_x=0.5)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        #fig.update_xaxes(tickangle=90,tickfont=dict( size=9))
        fig.update_layout(template='plotly_white',width=900,height=550)
        return(fig)

def plot_countries_weekly(countries=['UK','US'],startDate="2020-01-21",endDate='2020-01-21',
                          weekly=False, title="Covid19 cases over weeks",colsu=2):

    if (len(countries)==0):
        countries = ['US']
    temp = covid19[covid19.Country.isin(countries)].copy()
    
    temp['Date'] = pd.to_datetime(temp['Date'])

    mask = (temp['Date'] >= pd.to_datetime(startDate))&(temp['Date'] <= pd.to_datetime(endDate))
    temp = temp.loc[mask]
    
    temp_grouped = temp[temp.Country.isin(countries)].groupby(['Country',temp['Date'].dt.strftime('%W')])['Confirmed','Deaths','Recovered'].max().reset_index()
    tmpM = pd.melt(temp_grouped, id_vars=['Country','Date'],\
             value_vars=['Confirmed','Deaths','Recovered'], var_name='Week', value_name='Total')
    tmpM.columns = ['Country','Week','Cases','Total']
    #tmpM['Date'] = tmpM['Date'].dt.strftime('%d-%m-%Y')

    
    if (weekly):
        fig = px.line(tmpM[tmpM.Country.isin(countries)], x='Week', y='Total', color='Cases',\
                     template='plotly_white',facet_col='Country', facet_col_wrap=colsu)
        title = 'Covid19 cases' + ' between '+str(startDate)+ ' and '+str(endDate)

        fig.update_layout(title_text=title, title_x=0.5)
        fig.update_layout(xaxis_title="Week No")

        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    else:

        temp = df_covid19[df_covid19.Country.isin(countries)].copy()
    
        temp['Date'] = pd.to_datetime(temp['Date'])
        mask = (temp['Date'] >= pd.to_datetime(startDate))&(temp['Date'] <= pd.to_datetime(endDate))
        temp = temp.loc[mask].copy()
    
        x = temp.groupby(['Country','Cases', 'Date'], as_index=False).agg({'Count':'sum'})

        x['Date'] = pd.to_datetime(x['Date'])
        #x['Date'] = x['Date'].dt.strftime('%d-%m-%Y')

        #xg = x.groupby(['Country', x['Date'].dt.strftime('%W')])['Count'].max().reset_index()

        x.head()
        title = 'Covid19 cases' + ' between '+str(startDate)+ ' and '+str(endDate)
        fig = px.line(x[x.Country.isin(countries)], x='Date', y='Count', color='Cases',\
                     template='plotly_white',facet_col='Country', facet_col_wrap=colsu)

        fig.update_layout(title_text=title, title_x=0.5)
        #fig.update_layout(xaxis={'tickformat':"%b %d %Y "
        #        ,'type':'category'
        #})
        
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(template='plotly_white',width=900,height=550)

    return (fig)




# plot one or more country in one plot 
def plot_countries_oneD(countries=[],weekly=True, cases='Confirmed',start_date=start_date_df,
    end_date=end_date_df):

    temp = df_covid19.copy()
    temp['Date'] = pd.to_datetime(temp['Date'])

    #temp['Daily']=0
    mask = (temp['Date'] > pd.to_datetime(start_date))&(temp['Date'] <= pd.to_datetime(end_date))
    temp = temp.loc[mask]
    temp.sort_values(by=['Country', 'Date'])


    if weekly==True: 
        temp_grouped = temp[temp.Cases==cases].groupby(['Country', temp['Date'].dt.strftime('%W')])['Count'].max().reset_index()
        temp_grouped.columns = ['Country','Week','Cases']

        temp_grouped.head(2)



        fig = go.Figure()
        for i in range(len(countries)):
            tmp = temp_grouped[temp_grouped.Country==countries[i]]
            fig.add_trace(go.Scatter(
                x=tmp['Week'],
                y=tmp['Cases'],
                name=cases+ " Cases ("+str(countries[i])+")",
                mode='lines+markers'
            ))


        #fig.update_layout(barmode='group')
        fig.update_layout(template='plotly_white')
        fig.update_layout(title_text='Number of '+cases+" cases over weeks "+'(cumulative)', title_x=0.5,
            xaxis_title="Week Number of 2020 (Week 3 = 21st of Jan,...)",
            yaxis_title="Number of " + cases + " Cases",)
    elif weekly==False: 
        temp_grouped = temp[temp.Cases==cases].groupby(['Country','Date'])['Count'].max().reset_index()
        temp_grouped.columns = ['Country','Date','Cases']

        temp_grouped.head(2)



        fig = go.Figure()
        for i in range(len(countries)):
            tmp = temp_grouped[temp_grouped.Country==countries[i]]
            fig.add_trace(go.Scatter(
                x=tmp['Date'],
                y=tmp['Cases'],
                name=cases+ " Cases ("+str(countries[i])+")",
                mode='lines+markers'
            ))


        #fig.update_layout(barmode='group')
        fig.update_layout(title_text='Number of ' +cases+" cases over time ", title_x=0.5,
            xaxis_title="Date",
            yaxis_title="Number of " + cases + " Cases" + '(cumulative)',)
      
    fig.update_layout(template='plotly_white',width=1100,height=550)

    return fig

# SHOULD update subject to the case type (con, death, recovered)
def plot_specific_country(startDate,endDate,bar=True,line=False,log=False):


    if daily_spread_country: 
        if covid19_cases=='Confirmed':
                # selection box won't work if the list is too long 
                if regions == 'Worldwide':
                    country = st.sidebar.selectbox('',top_conf)
                    st.plotly_chart(plot_countries_daily([country],
                    covid19_cases,startDate,endDate,
                                 ' Covid19 daily ',4,plot_bar,plot_line,log))
                if regions =='Europe':

                    country = st.sidebar.selectbox('',top_conf_eus)
                    st.plotly_chart(plot_countries_daily([country],
                    covid19_cases,startDate,endDate,
                                 ' Covid19 daily ',4,plot_bar,plot_line,log))
                if regions =='Arab World':
                    country = st.sidebar.selectbox('',top_conf_arabs)
                    st.plotly_chart(plot_countries_daily([country],
                    covid19_cases,startDate,endDate,
                                 ' Covid19 daily ',4,plot_bar,plot_line,log))
        if covid19_cases=='Deaths':
                # selection box won't work if the list is too long 
                if regions == 'Worldwide':
                    country = st.sidebar.selectbox('',top_conf)
                    st.plotly_chart(plot_countries_daily([country],
                    covid19_cases,startDate,endDate,
                                 ' Covid19 daily ',4,plot_bar,plot_line,log))
                if regions =='Europe':

                    country = st.sidebar.selectbox('',top_death_eus)
                    st.plotly_chart(plot_countries_daily([country],
                    covid19_cases,startDate,endDate,
                                 ' Covid19 daily ',4,plot_bar,plot_line,log))
                if regions =='Arab World':
                    country = st.sidebar.selectbox('',top_death_arabs)
                    st.plotly_chart(plot_countries_daily([country],
                    covid19_cases,startDate,endDate,
                                 ' Covid19 daily ',4,plot_bar,plot_line,log))

        if covid19_cases=='Recovered':
                # selection box won't work if the list is too long 
                if regions == 'Worldwide':
                    country = st.sidebar.selectbox('',top_conf)
                    st.plotly_chart(plot_countries_daily([country],
                    covid19_cases,startDate,endDate,
                                 ' Covid19 daily ',4,plot_bar,plot_line,log))
                if regions =='Europe':

                    country = st.sidebar.selectbox('',top_rec_eus)
                    st.plotly_chart(plot_countries_daily([country],
                    covid19_cases,startDate,endDate,
                                 ' Covid19 daily ',4,plot_bar,plot_line,log))
                if regions =='Arab World':
                    country = st.sidebar.selectbox('',top_rec_arabs)
                    st.plotly_chart(plot_countries_daily([country],
                    covid19_cases,startDate,endDate,
                                 ' Covid19 daily ',4,plot_bar,plot_line,log))

def plot_countries_all(startDate="2020-01-21",endDate='2020-01-21',
                           title="Covid19 cases over weeks",bar=True):

    
    temp = covid19.copy()
    
    temp['Date'] = pd.to_datetime(temp['Date'])

    mask = (temp['Date'] >= pd.to_datetime(startDate))&(temp['Date'] <= pd.to_datetime(endDate))
    temp = temp.loc[mask]
    
    temp_grouped = temp.groupby([temp['Date'].dt.strftime('%W')])['Confirmed','Deaths','Recovered'].max().reset_index()
    tmpM = pd.melt(temp_grouped, id_vars=['Date'],\
             value_vars=['Confirmed','Deaths','Recovered'], var_name='Week', value_name='Total')
    tmpM.columns = ['Week','Cases','Total']
    #tmpM['Date'] = tmpM['Date'].dt.strftime('%d-%m-%Y')

    
    

    temp = df_covid19.copy()
    
    temp['Date'] = pd.to_datetime(temp['Date'])
    mask = (temp['Date'] >= pd.to_datetime(startDate))&(temp['Date'] <= pd.to_datetime(endDate))
    temp = temp.loc[mask].copy()
    
    x = temp.groupby(['Cases', 'Date'], as_index=False).agg({'Count':'sum'})

    x['Date'] = pd.to_datetime(x['Date'])
        #x['Date'] = x['Date'].dt.strftime('%d-%m-%Y')

        #xg = x.groupby(['Country', x['Date'].dt.strftime('%W')])['Count'].max().reset_index()

    x.head()
    title = 'Covid19 cases' + ' between '+str(startDate)+ ' and '+str(endDate)
    if bar: 
        fig = px.bar(x, x='Date', y='Count', color='Cases',\
                     template='plotly_white')
    else:
        fig = px.line(x, x='Date', y='Count', color='Cases',\
                     template='plotly_white')
    fig.update_layout(title_text=title, title_x=0.5)
        #fig.update_layout(xaxis={'tickformat':"%b %d %Y "
        #        ,'type':'category'
        #})
        
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(template='plotly_white',width=900,height=550)

    return (fig)

if daily_spread:
    plot_countries_daily_s(start_date,end_date,n)

show_totals = st.sidebar.checkbox('Show covid19 Cases/ Globally')

if show_totals:
    fig = plot_countries_all(start_date,end_date,
                      'Covid19 Weekly/Daily spread (commulative) ',plot_bar)
    st.plotly_chart(fig)


countries_cumm_sums = st.sidebar.checkbox('Total Numbers',0)
            #st.write(str(top_conf))
if countries_cumm_sums:
    countries_m = st.sidebar.multiselect('Select Country/s', top_conf, default=top_conf[:1])
    weekly = st.sidebar.radio("Weekly?", 
    ['Weeks','Days'])
    weeks = (weekly=='Weeks')
    fig = plot_countries_weekly(countries_m,start_date,end_date,weeks,
                      'Covid19 weekly spread (commulative) ',3)
    st.plotly_chart(fig)
# SHOU

daily_spread_country = st.sidebar.checkbox('Daily Spread (Specific Country)')
plot_specific_country(start_date,end_date,plot_bar,plot_line,log_plot)

# one or more country in the same plot 
#plot_countries_oneD(countries=[],weekly=True, cases='Confirmed',start_date=start_date_df,
# compare countries 

compare_countries_by_cases = st.sidebar.checkbox('Compare Countries Numbers',0,'uniquecomparecountriesnumber')
if compare_countries_by_cases:
    countries_m = st.sidebar.multiselect('Select one or more Country', top_conf, default=top_conf[:1],key=12345690)
    weekly = st.sidebar.radio("Weekly?", 
    ['Weeks','Days'],key=123)
    weeks = (weekly=='Weeks')
    fig = plot_countries_oneD(countries_m,weeks,covid19_cases,start_date,end_date)
    st.plotly_chart(fig)

show_tables = st.checkbox('Show List of Countries',)
if show_tables:
    st.markdown('List of top countries by number casesthe')
    show_top_countries_list(number_s)


#33
# dlkj 
# px.set_mapbox_access_token(secret_value_0)
# fig = px.scatter_mapbox(df_hotspots,
#                         lat="latitude",
#                         lon="longitude",
#                         size="confirmed",
#                         hover_data=['infection_case','city','province'],
#                         zoom=5,
#                         size_max=50,
#                         title= 'COVID19 Hotspots in South Korea')
# fig.show()



# df = df_plot.query("Country_Region=='India'")
# df.reset_index(inplace = True)
# df = add_daily_measures(df)
# fig = go.Figure(data=[
#     go.Bar(name='Cases', x=df['Date'], y=df['Daily Cases']),
#     go.Bar(name='Deaths', x=df['Date'], y=df['Daily Deaths'])
# ])
# # Change the bar mode
# fig.update_layout(barmode='overlay', title='Daily Case and Death count(India)',
#                  annotations=[dict(x='2020-03-23', y=106, xref="x", yref="y", text="Lockdown Imposed(23rd March)", showarrow=True, arrowhead=1, ax=-100, ay=-100)])
# fig.show()
