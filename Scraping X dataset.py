#!/usr/bin/env python
# coding: utf-8

# ## We are using TSA checkpoint travel numbers as X dataset

# In[35]:


# Import libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime


# In[36]:


# URL object
url = 'https://www.tsa.gov/coronavirus/passenger-throughput'
# page object
page = requests.get(url)


# In[37]:


page


# #### Response 200
# We can collect data from this url

# In[38]:


soup = BeautifulSoup(page.text, 'lxml')


# In[39]:


# Extract information from tag <table>
table = soup.find('table')


# In[40]:


# Extract headers from the web page
headers = []
for i in table.find_all('th'):
    title = i.text
    headers.append(title)
    
print("Columns:", headers)


# In[41]:


passenger_travel_df = pd.DataFrame(columns = headers)


# In[42]:


# Extract other information from the table
for j in table.find_all('tr')[1:]:
    row_data = j.find_all('td')
    row = [i.text for i in row_data]
    length = len(passenger_travel_df)
    passenger_travel_df.loc[length] = row


# In[43]:


passenger_travel_df.shape


# In[44]:


passenger_travel_df.tail()


# In[45]:


passenger_travel_df['Date'] = pd.to_datetime(passenger_travel_df['Date'])


# ### This dataset contains data from 2019 to 2022 till 5/15/2022

# In[46]:


passenger_travel_df['2022'] = passenger_travel_df['2022'].str.replace(",", "").replace(" ", "")
passenger_travel_df['2021'] = passenger_travel_df['2021'].str.replace(",", "").replace(" ", "")
passenger_travel_df['2020'] = passenger_travel_df['2020'].str.replace(",", "").replace(" ", "")
passenger_travel_df['2019'] = passenger_travel_df['2019'].str.replace(",", "").replace(" ", "")


# In[47]:


# passenger_travel_df.head(25)


# In[48]:


# passenger_travel_df.to_csv('passenger_travel_data.csv', index=False)


# In[49]:


df = pd.read_csv('passenger_travel_data.csv')
df = df.sort_values(by=['Date'])
df['date_old'] = df['Date']
# df


# In[50]:


def get_values(row):
#     print(row)
    if row['year'] == '2022':
        value = row['2022']
    elif row['year'] == '2021':
        value = row['2021']
    elif row['year'] == '2020':
        value = row['2020']
    elif row['year'] == '2019':
        value = row['2019']   
    return value


# In[51]:


df1 = df.copy()
df1['Date'] = pd.to_datetime(df1['Date'])
df1['Date'] = pd.to_datetime(df1['Date'].dt.strftime('2022-%m-%d'))
df1['year'] = df1['Date'].dt.strftime('%Y')
df1['travel_count'] = df1.apply(lambda row: get_values(row), axis=1)
df1.drop(['date_old', '2022', '2021', '2020', '2019', 'year'], axis=1, inplace=True)
df1 = df1.sort_values(by=['Date'])
# df1


# In[52]:


df2 = df.copy()
df2['Date'] = pd.to_datetime(df2['Date'])
df2['Date'] = pd.to_datetime(df2['Date'].dt.strftime('2021-%m-%d'))
df2['year'] = df2['Date'].dt.strftime('%Y')
df2['travel_count'] = df2.apply(lambda row: get_values(row), axis=1)
df2.drop(['date_old', '2022', '2021', '2020', '2019', 'year'], axis=1, inplace=True)
df2 = df2.sort_values(by=['Date'])
# df2


# In[53]:


df3 = df.copy()
df3['Date'] = pd.to_datetime(df3['Date'])
df3['Date'] = pd.to_datetime(df3['Date'].dt.strftime('2020-%m-%d'))
df3['year'] = df3['Date'].dt.strftime('%Y')
df3['travel_count'] = df3.apply(lambda row: get_values(row), axis=1)
df3.drop(['date_old', '2022', '2021', '2020', '2019', 'year'], axis=1, inplace=True)
df3 = df3.sort_values(by=['Date'])
# df3


# In[54]:


df4 = df.copy()
df4['Date'] = pd.to_datetime(df4['Date'])
df4['Date'] = pd.to_datetime(df4['Date'].dt.strftime('2019-%m-%d'))
df4['year'] = df4['Date'].dt.strftime('%Y')
df4['travel_count'] = df4.apply(lambda row: get_values(row), axis=1)
df4.drop(['date_old', '2022', '2021', '2020', '2019', 'year'], axis=1, inplace=True)
df4 = df4.sort_values(by=['Date'])
# df4


# ### Combining all the years to get two columns: Date and Travel count

# In[55]:


df_combined = df4.append([df3, df2, df1], ignore_index=True)
df_combined.set_index('Date', inplace=True)
df_combined.index = pd.to_datetime(df_combined.index)
df_combined


# ### This travel dataset does not contain value for 16th May every year. So those 4 values are missing

# In[56]:


df_combined.to_csv('daily_passenger_travel_data.csv', index=True)


# ### Combining X data with Covid cases and vaccines dataset

# In[59]:


cleaned_cases_df = pd.read_csv('covid_cases_and_death_all_states_with_outlier_removal.csv')
cleaned_cases_df = cleaned_cases_df.sort_values(by=['date'])
cleaned_cases_df['Date'] = cleaned_cases_df['date']
cleaned_cases_df.drop(['date'], axis=1, inplace=True)
cleaned_cases_df.set_index('Date', inplace=True)
cleaned_cases_df


# In[60]:


cleaned_cases_df = cleaned_cases_df.groupby('Date').sum()
# cleaned_cases_df.drop(['total_cases_cum', 'total_death_cum'], axis=1, inplace=True)
# cleaned_cases_df['Date'] = pd.to_datetime(cleaned_cases_df['Date'])
cleaned_cases_df.index = pd.to_datetime(cleaned_cases_df.index)
cleaned_cases_df


# In[61]:


cleaned_vaccine_df = pd.read_csv('vaccination_all_states_cleaned.csv')
cleaned_vaccine_df['Date'] = pd.to_datetime(cleaned_vaccine_df['Date'])
cleaned_vaccine_df = cleaned_vaccine_df.sort_values(by=['Date'])
cleaned_vaccine_df.set_index('Date', inplace=True)
cleaned_vaccine_df.drop(['Unnamed: 0'], axis=1, inplace=True)
cleaned_vaccine_df


# In[62]:


cleaned_vaccine_df = cleaned_vaccine_df.groupby('Date').sum()
cleaned_vaccine_df.index = pd.to_datetime(cleaned_vaccine_df.index)
cleaned_vaccine_df


# In[63]:


# df_combined

# merged_df = pd.merge(cleaned_cases_df, cleaned_vaccine_df, on="Date")
merged_df = pd.merge(cleaned_cases_df, df_combined, on="Date")
merged_df


# In[64]:


merged_df.to_csv('cases_and_passenger_travel_data.csv', index=True)


# ### Uncleaned data

# In[65]:


uncleaned_cases_df = pd.read_csv('covid_cases_and_death_all_states_without_outlier_removal.csv')
uncleaned_cases_df = uncleaned_cases_df.sort_values(by=['date'])
uncleaned_cases_df['Date'] = uncleaned_cases_df['date']
uncleaned_cases_df.drop(['date'], axis=1, inplace=True)
uncleaned_cases_df.set_index('Date', inplace=True)
uncleaned_cases_df


# In[66]:


uncleaned_cases_df = uncleaned_cases_df.groupby('Date').sum()
uncleaned_cases_df.index = pd.to_datetime(uncleaned_cases_df.index)
uncleaned_cases_df.drop(['total_cases_cum', 'total_death_cum'], axis=1, inplace=True)
uncleaned_cases_df


# In[67]:


uncleaned_vaccine_df = pd.read_csv('vaccination_all_states.csv')
uncleaned_vaccine_df['Date'] = pd.to_datetime(uncleaned_vaccine_df['Date'])
uncleaned_vaccine_df = uncleaned_vaccine_df.sort_values(by=['Date'])
uncleaned_vaccine_df.set_index('Date', inplace=True)
uncleaned_vaccine_df.drop(['Unnamed: 0'], axis=1, inplace=True)
uncleaned_vaccine_df


# In[68]:


uncleaned_vaccine_df = uncleaned_vaccine_df.groupby('Date').sum()
uncleaned_vaccine_df.index = pd.to_datetime(uncleaned_vaccine_df.index)
uncleaned_vaccine_df


# In[69]:


# df_combined

# merged_clean_df = pd.merge(uncleaned_cases_df, uncleaned_vaccine_df, on="Date")
merged_clean_df = pd.merge(uncleaned_cases_df, df_combined, on="Date")
merged_clean_df


# In[70]:


merged_clean_df.to_csv('clean_cases_and_passenger_travel_data.csv', index=True)

