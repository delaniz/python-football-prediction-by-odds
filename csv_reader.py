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

loc = 'c:/Users/DenizKaldi/python/data/'
league = "all-euro-data-"
columns_req = ["Div","HomeTeam","AwayTeam","HTHG","HTAG","HTR","FTHG","FTAG","FTR", \
            "HS","AS","HST","AST","HF","AF","HC", "Date","AC","HY", \
            "AY","HR","AR","B365H","B365D","B365A","BWH","BWD","BWA","BbAv>2.5","BbAv<2.5"]

#columns_new =columns_req - ['AR', 'AS', 'HY', 'AF', 'HR', 'HST', 'HS', 'HC', 'AY', 'AC', 'HF', 'AST']
            
#"result_1x2","result_overunder","result1_asianhandicap","avg_winH","avg_winD","avg_winA","avg_ah_1x","avg_ah_x2"
all_matches = pd.DataFrame()

""" READ DATA BETWEEN 2005 - 2009 """
"""for fileName in glob.glob(loc+league+"m-[0][5-9][0][0-9]*.csv"): #2005 - 2009 
    print(fileName)   
    actualFile = pd.read_csv(fileName,usecols=columns_req, sep=',', error_bad_lines=False, index_col=False,engine='python')
    all_matches = all_matches.append(actualFile,ignore_index = True) # join='outer', axis=1).fillna(0)
    print('readen')


league_1920_E = [col.replace('BbAv<2.5','Avg<2.5').replace('BbAv>2.5','Avg>2.5') for col in columns_req]
league_1619_EC = list((Counter(columns_req)-Counter(['HC', 'HS', 'HST', 'AC', 'HF', 'AF', 'AS', 'AST'])).elements())
league_1920_EC = [col.replace('BbAv<2.5','Avg<2.5').replace('BbAv>2.5','Avg>2.5') for col in league_1619_EC] 

#print(league_1920_E)
#print(re.search('.*1[019]2[0-9]-E.*','englandm-1920-E0.csv'))
"""
""" READ DATA BETWEEN 2009 - 2021 """
"""for fileName in glob.glob(loc+league+"m-[1-2][0-9][1][8-9]*.csv"): #2005 - 2009  #[1-2][0-9][1-2][0-9]*
    if re.search('.*17|18|19|20.*',fileName):
        print(fileName)
        actual_columns = league_1619_EC if re.search('.*1[6-8]1[7-9]-EC.*',fileName) \
                        else league_1920_EC if re.search('.*[12][019]2[0-9]-EC.*',fileName) \
                        else league_1920_E if re.search('.*[12][019]2[0-9]-E.*',fileName) else columns_req
        #print(actual_columns)
        actualFile = pd.read_csv(fileName,usecols=actual_columns, sep=',', error_bad_lines=False,index_col=False,engine='python')
        dateFormat = '%d/%m/%Y' if (re.search('.*[12][7][12][0-9]-E0.*',fileName) or \
                                    re.search('.*[1][8-9][12][8-9]-E.*',fileName) or \
                                    re.search('.*[1][9][2][0]-E.*',fileName) or \
                                    re.search('.*[2][0-9][12][0-9]-E.*',fileName) ) \
                                else '%d/%m/%y'  
        print(dateFormat)
        actualFile = pd.read_csv(fileName,usecols=actual_columns,parse_dates=['Date'], sep=',', error_bad_lines=False,index_col=False,
                        date_parser=lambda x: pd.to_datetime(x, format=try_parsing_date(x)),engine='python')
        if re.search('.*[12][019]2[0-9]-E.*',fileName):
            all_matches.rename(columns = {'BbAv<2.5':'Avg<2.5','BbAv>2.5':'Avg>2.5'}, inplace = True)   
        all_matches = all_matches.append(actualFile,ignore_index = True).fillna(0)
        print('readen')


"""
#def getNewLeague(league):
all_matches_new = pd.DataFrame()
columns_new_league = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR','PSH', 'PSD', 'PSA',"AvgH","AvgD","AvgA"]
for fileName in glob.glob("c:/Users/DenizKaldi/python/data/*.csv"):
    print(fileName)
    actualFile = pd.read_csv(fileName,sep=',', error_bad_lines=False, index_col=False,engine='python',parse_dates=['Date'],
                                date_parser=lambda x: pd.to_datetime(x, format=try_parsing_date(x)))
    if 'Country' not in actualFile.columns:
        actualFile['Country'] = ''
    actualFile.rename(columns = {'League':'Div','Home':'HomeTeam','Away':'AwayTeam','HG':'FTHG','AG':'FTAG','Res':'FTR',
                            'PH':'PSH','PD':'PSD','PA':'PSA'}, inplace = True) 
    actualFile['Div'] = actualFile.apply(lambda x: x['Country'][0:3] if x['Country']!='' else x['Div'],axis=1)
    actualFile= actualFile[columns_new_league]
    print(actualFile)
    print(actualFile[actualFile['Date']=='Superliga'])
    all_matches_new = pd.concat([actualFile,all_matches_new],  axis=0, ignore_index=True).fillna(0)
    print(all_matches_new)
    
    print('readen')



#all_matches_new['Div'] = all_matches_new.apply(lambda x: x['Country'][0:3] if x['Country']!=0 else x['Div'],axis=1)
#all_matches_new = all_matches_new.drop(['Country','MaxH','MaxD','MaxA','Season','Time'],axis=1)
#all_matches['Div'] =all_matches['Div']
all_matches_new.sort_values(by='Date',ascending=False,inplace=True)
print(all_matches_new.head(10))
print(all_matches_new.tail(10))
#7/0
#print(all_matches_new.columns)
print(all_matches_new[all_matches_new['Date']=='Superliga'])

columns = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST',  
             'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 
            'IWA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA', 'PSCH', 'PSCD', 'PSCA',"BbAv>2.5",
            "BbAv<2.5","BbAvH","BbAvD","BbAvA",'BbAvAHH', 'BbAvAHA' ]
columns_new = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST',  
             'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 
            'IWA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA', 'PSCH', 'PSCD', 'PSCA',"Avg>2.5",
            "Avg<2.5","AvgH","AvgD","AvgA",'AvgAHH', 'AvgAHA' ]
columns_new2 = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR','PSH', 'PSD', 'PSA',"AvgH","AvgD","AvgA", 'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST',  
             'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 
            'IWA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA', 'PSCH', 'PSCD', 'PSCA',"Avg>2.5",
            "Avg<2.5",'AvgAHH', 'AvgAHA',"BbAv>2.5",
            "BbAv<2.5","BbAvH","BbAvD","BbAvA",'BbAvAHH', 'BbAvAHA' ]
columns_diff = ['Bb1X2', 'BbMxH', 'BbAvH', 
            'BbMxD', 'BbAvD', 'BbMxA', 'BbAvA', 'BbOU', 'BbMx>2.5', 'BbAv>2.5', 'BbMx<2.5', 'BbAv<2.5', 'BbAH', 'BbAHh', 'BbMxAHH', 
             'BbAvAHH', 'BbMxAHA', 'BbAvAHA']
columns_1920 = list((Counter(columns)-Counter(columns_diff)).elements())
for fileName in glob.glob(loc+league+'*.xlsx'):
    #columns_to_read = columns_1920 if re.search('.*2020.*',fileName) else columns
      
    print(fileName)
    season_matches = pd.read_excel(fileName, sheet_name=None)
    season_matches = pd.concat(season_matches, axis=0, ignore_index=True)
    #season_matches['Country'] = ''
    if re.search('.*2020.*',fileName):
        columns = columns_new
    season_matches = season_matches[columns]
    #print(season_matches)
    
    all_matches = pd.concat([season_matches,all_matches],axis=0,ignore_index=True).fillna(0)
    #print(all_matches.columns)
    #print(all_matches)
    
#print(all_matches.shape)
#print(pd.DataFrame(all_matches.iloc[1]))
#print(all_matches.columns)
#print(all_matches)
#print(all_matches_new)

all_matches = all_matches[columns_new2]   #'PSH','PSD', 'PSA', 'AvgH', 'AvgD', 'AvgA']
all_matches = pd.concat([all_matches,all_matches_new],axis=0,join='outer',ignore_index=True).fillna(0)

#print(all_matches)
#print(all_matches.columns)
#print(all_matches[all_matches['Date']=='Superliga'])

#all_matches = all_matches[all_matches['HomeTeam']!=0]
all_matches['AvgH'] = all_matches.apply(lambda x: x['BbAvH'] if x['AvgH'] == 0 else x['AvgH'],axis=1)
all_matches['AvgD'] = all_matches.apply(lambda x: x['BbAvD'] if x['AvgD'] == 0 else x['AvgD'],axis=1)
all_matches['AvgA'] = all_matches.apply(lambda x: x['BbAvA'] if x['AvgA'] == 0 else x['AvgA'],axis=1)
all_matches['Avg>2.5'] = all_matches.apply(lambda x: x['BbAv>2.5'] if x['Avg>2.5'] == 0 else x['Avg>2.5'],axis=1)
all_matches['Avg<2.5'] = all_matches.apply(lambda x: x['BbAv<2.5'] if x['Avg<2.5'] == 0 else x['Avg<2.5'],axis=1)
all_matches['AvgAHH'] = all_matches.apply(lambda x: x['BbAvAHH'] if x['AvgAHH'] == 0 else x['AvgAHH'],axis=1)
all_matches['AvgAHA'] = all_matches.apply(lambda x: x['BbAvAHA'] if x['AvgAHA'] == 0 else x['AvgAHA'],axis=1)
#all_matches.drop(['BbAv>2.5', 'BbAv<2.5', 'BbAvH', 'BbAvD',   'BbAvA', 'BbAvAHH', 'BbAvAHA'],axis=1)
all_matches = all_matches[columns_new]
print(all_matches.columns)
print(all_matches.describe)

#import pickle
#all_matches.to_pickle('all_1721_now')
#all_matches.to_csv('all_1721_now.csv', encoding='utf-8',index=False)
#print(all_matches['Date'].unique())
print(all_matches[all_matches['Date']=='Superliga'])

#all_matches = pd.concat([all_matches_main,all_matches_new],join='outer',axis=1).fillna(0)
all_matches['Date'] = all_matches['Date'].apply(lambda x: pd.Timestamp(x, tz=None).to_pydatetime())
#print(all_matches.columns)
#all_matches= all_matches[all_matches['Div']=='E1']
print(all_matches.describe())
print(all_matches.head(10))
print(all_matches.tail(10))
print(all_matches['Div'].unique())
#print(all_matches['Country'].unique())
#7/0
print('######################')
all_matches.sort_values(by='Date',ascending=False,inplace=True)
print(all_matches.head(10))
print(all_matches.tail(10))

#7/0

print('///////////////////')
all_matches['Year']         = all_matches['Date'].apply(lambda x: x.year)
all_matches['Month']        = all_matches['Date'].apply(lambda x: x.month)
all_matches['Day']          = all_matches['Date'].apply(lambda x: x.day )

# REMOVE BLANK ROWS 
#all_matches = all_matches[all_matches['Year']>=2005]

all_matches.sort_values(by=['Year','Month','Day'],ascending=False,inplace=True)
print(all_matches.head(10))
print(all_matches.tail(10))



all_matches['HTGDiff']      = all_matches['FTHG'] - all_matches['FTAG']
all_matches['ATGDiff']      = all_matches['FTAG'] - all_matches['FTHG']
"""all_matches['AvgH']         = (all_matches['B365H']+all_matches['BWH']) / 2
all_matches['AvgA']         = (all_matches['B365A']+all_matches['BWA']) / 2
all_matches['AvgD']         = (all_matches['B365D']+all_matches['BWD']) / 2
"""

all_matches['Winner'] = np.where(all_matches['FTR']=='H', all_matches['HomeTeam'],np.where(all_matches['FTR']=='A',all_matches['AwayTeam'],''))
all_matches['Loser'] = np.where(all_matches['FTR']=='A', all_matches['HomeTeam'],np.where(all_matches['FTR']=='H',all_matches['AwayTeam'],''))
#part = all_matches[all_matches['Div']=='E0']
#print(part.head(5))
#print(part.tail(5))
#print('---above E0 leage data')


#print(all_matches['AvgH'].isnull().sum())
all_matches=all_matches[['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST',  
             'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 
            'IWA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA', 'PSCH', 'PSCD', 'PSCA',"Avg>2.5",
            "Avg<2.5","AvgH","AvgD","AvgA",'AvgAHH', 'AvgAHA',"Year","Month","Day","HTGDiff","ATGDiff","Winner","Loser"]]


print(all_matches.head(5))
print(all_matches.tail(5))
print(all_matches.describe())
print(all_matches.columns)
#print(all_matches[all_matches['AvgH']==0])

import pickle
all_matches.to_pickle('all_data_2017_2021')
all_matches.to_csv('all_data_2017_2021.csv', encoding='utf-8',index=False)

#print(all_matches.describe())
#missing_part1 = pd.read_csv('c:/Users/DenizKaldi/Documents/wetten/csv/englandm-1718-E0.csv',sep=',', error_bad_lines=False, index_col=False)
#all_matfhes = pd.concat([all_matches,missing_part1], join='outer', axis=1).fillna(0)