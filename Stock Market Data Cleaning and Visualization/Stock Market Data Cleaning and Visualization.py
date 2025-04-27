# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:05:45 2024

@author: j
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from datetime import datetime
import re  # Import the re module for regular expressions

#We Read the CSV file into a DataFrame
df = pd.read_csv('data_with_indicators1.csv')

# Print the first few rows of the DataFrame
print(df.head())

#The code print information about the DataFrame
print(df.info())

#We print summary statistics about the DataFrame
print(df.describe())

# Define columns to fill with median values
columns_to_fill = ['Rtn..1.Day', 'Rtn..1.Day.1', 'Rtn..2.Day']

# Fill missing values in the specified columns with their median values
for column in columns_to_fill:
    median_value = df[column].median()
    df[column].fillna(median_value, inplace=True)

# Print updated information about the DataFrame
print(df.info())
print(df.describe())

# Create a summary DataFrame by grouping the original DataFrame by certain columns
summary = df.groupby(['Date', 'Rtn..1.Day', 'Rtn..1.Day.1', 'Rtn..2.Day']).size().reset_index(name='count')
print(summary.head())
print(summary.info())


# Reset the index of the summary DataFrame
summary.reset_index(drop=True, inplace=True)
print(summary.head())
print(summary.info())


#We sample a date string
s = '2000/06/12'

#We compile regular expression to replace slashes with hyphens
pattern = re.compile(r'[/.-]')

#Replace slashes in the sample date string with hyphens
formatted_date = pattern.sub('-', s)
print('Dates changed to the new format:', formatted_date)

#Replace slashes in the 'Date' column of the DataFrame with hyphens
df['Date'] = df['Date'].astype(str).apply(lambda x: pattern.sub('-', x))

#Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

#Replace slashes in the 'Date' column of the summary DataFrame with hyphens
summary['Date'] = summary['Date'].astype(str).apply(lambda x: pattern.sub('-', x))

#Here we convert the 'Date' column of the summary DataFrame to datetime format
summary['Date'] = pd.to_datetime(summary['Date'], format='%Y-%m-%d')


#We extract year, month, and day from the 'Date' column
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

#we print the updated DataFrame
print(df.head())


# Create a count plot of shares per year
plt.figure(figsize=(12, 6))
sns.countplot(x= 'Year', data=df)
plt.ylim(240, 255)
plt.title('Total Shares per year')
plt.xlabel('Year')
plt.ylabel('Total Shares invested for the year')
plt.show()


# Create a line plot of share indicators
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Date', y='Rtn..2.Day', label='Rtn..2.Day', color='Yellow')
sns.lineplot(data=df, x='Date', y='Rtn..1.Day.1', label='Rtn..1.Day.1', color='blue')
plt.title('Shares Values per Record')
plt.xlabel('Date')  # Changed from 'Record number' to 'Date'
plt.ylabel('Share Value')
plt.legend()
plt.show()