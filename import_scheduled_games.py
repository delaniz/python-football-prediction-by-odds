import pandas as pd
import glob
import codecs
import numpy as np
from collections import Counter
import re
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from datetime import datetime

def try_parsing_date(text):
    for fmt in ('%d/%m/%Y','%d/%m/%y','%Y-%m-%d', '%d.%m.%Y'):
        try:
            datetime.strptime(text, fmt)
            return fmt
        except ValueError:
            pass
    raise ValueError('no valid date format found')

columns_main = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'PSH', 'PSD', 'PSA',"AvgH","AvgD","AvgA",'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 
            'IWA',  'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA', 'PSCH', 'PSCD', 'PSCA',"Avg>2.5",
            "Avg<2.5",'AvgAHH', 'AvgAHA' ]
            
columns_other = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'PSH', 'PSD', 'PSA',"AvgH","AvgD","AvgA"]
columns_order = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 
            'IWA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA', 'PSCH', 'PSCD', 'PSCA',"Avg>2.5",
            "Avg<2.5","AvgH","AvgD","AvgA",'AvgAHH', 'AvgAHA','Year','Month','Day' ]
target_columns = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST',  
             'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 
            'IWA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA', 'PSCH', 'PSCD', 'PSCA',"Avg>2.5",
            "Avg<2.5","AvgH","AvgD","AvgA",'AvgAHH', 'AvgAHA',"Year","Month","Day","HTGDiff","ATGDiff","Winner","Loser"]
columns_missing = list((Counter(target_columns)-Counter(columns_main)).elements())

loc = 'c:/Users/DenizKaldi/python/data/scheduled_games/*.csv'
all_matches_new = pd.DataFrame()

for fileName in glob.glob(loc):
    print(fileName)
    actualFile = pd.read_csv(fileName,sep=',', error_bad_lines=False, index_col=False,engine='python',parse_dates=['Date'],
                                date_parser=lambda x: pd.to_datetime(x, format=try_parsing_date(x)))
    if 'Country' in actualFile.columns:
        actualFile.rename(columns = {'League':'Div','Home':'HomeTeam','Away':'AwayTeam',
                            'PH':'PSH','PD':'PSD','PA':'PSA'}, inplace = True) 
        actualFile['Div'] = actualFile.apply(lambda x: x['Country'][0:3] if x['Country']!='' else x['Div'],axis=1)
        actualFile= actualFile[columns_other]
    else:
        actualFile = actualFile[columns_main]
    
    print(actualFile)
    #print(actualFile[actualFile['Date']=='Superliga'])
    all_matches_new = pd.concat([actualFile,all_matches_new],  axis=0, ignore_index=True).fillna(0)
    print(all_matches_new)
    print('readen')

all_matches_new['Date'] = all_matches_new['Date'].apply(lambda x: pd.Timestamp(x, tz=None).to_pydatetime())


for col in columns_missing:
    all_matches_new[col] = 0
#all_matches_new['Winner'] = ''
#all_matches_new['Loser'] = ''
all_matches_new['Year']         = all_matches_new['Date'].apply(lambda x: x.year)
all_matches_new['Month']        = all_matches_new['Date'].apply(lambda x: x.month)
all_matches_new['Day']          = all_matches_new['Date'].apply(lambda x: x.day )
all_matches_new = all_matches_new[target_columns]
all_matches_new.sort_values(by='Date',inplace=True,ascending=False)
print(all_matches_new[columns_other].tail(10))
print(all_matches_new[columns_other].head(100).to_string())

print(all_matches_new.columns)
print(all_matches_new.describe())

#print(all_matches_new['Div'].unique())



#to_csv('my_csv.csv', mode='a', header=False)

#import pickle
#all_matches_new.to_pickle('all_data_2017_2021')
all_matches_new = all_matches_new[all_matches_new['Date']>=pd.to_datetime('today').date()]
all_matches_new.to_csv('all_data_2017_2021.csv', mode='a', header=False,encoding='utf-8',index=False)

#print(all_matches.describe())
#missing_part1 = pd.read_csv('c:/Users/DenizKaldi/Documents/wetten/csv/englandm-1718-E0.csv',sep=',', error_bad_lines=False, index_col=False)
#all_matfhes = pd.concat([all_matches,missing_part1], join='outer', axis=1).fillna(0)