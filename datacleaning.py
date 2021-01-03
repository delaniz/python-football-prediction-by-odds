import pandas as pd
import glob
import codecs
import numpy as np
from collections import Counter
import re
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

loc = 'c:/Users/DenizKaldi/Documents/wetten/csv/'
league = "england"
columns_req = ["Div","HomeTeam","AwayTeam","HTHG","HTAG","HTR","FTHG","FTAG","FTR", \
            "HS","AS","HST","AST","HF","AF","HC", "Date","AC","HY", \
            "AY","HR","AR","B365H","B365D","B365A","BWH","BWD","BWA","BbAv>2.5","BbAv<2.5"]

            
#"result_1x2","result_overunder","result1_asianhandicap","avg_winH","avg_winD","avg_winA","avg_ah_1x","avg_ah_x2"
all_matches = pd.read_csv('england_2005_now.csv',usecols=columns_req, sep=',', error_bad_lines=False, index_col=False,engine='python')


print(all_matches.head())
print(all_matches.describe())
print(all_matches.columns)
#print(all_matches['AvgH'].isnull().sum())

time_mask = (all_matches['Date']>='2018.07.01') & (all_matches['Date']<'2019.07.01')
league_mask = (all_matches['Div'] =='E0') #for english premierleague
season_1819 = all_matches[ time_mask & league_mask]
season_1819.fillna(0)
avg_goal_diff(season_1819,'AvgHTGDiff','HomeTeam','H')
avg_goal_diff(season_1819,'AvgATGDiff','AwayTeam','A')
#season_1819['AvgFTHG']      = season_1819['AwayTeam'].apply()
#print(pd.DataFrame.from_dict(season_1819['AvgHTGDiff'].iloc[0],orient='index'))
season_1819.set_index = 'Date'
season_1819_n = season_1819[['Date','HomeTeam','AwayTeam','FTHG','FTAG','Avg>2.5','Avg<2.5','HTGDiff','ATGDiff','AvgHTGDiff','AvgATGDiff']]
season_1819_n.set_index = 'Date'
print(season_1819_n.describe())
print(season_1819_n.head())
scatter_matrix(season_1819_n, alpha=0.2, figsize=(10, 10))
plt.show()
all_matches.to_csv('england_2005_now.csv', encoding='utf-8',index=False)

#print(all_matches.describe())
#missing_part1 = pd.read_csv('c:/Users/DenizKaldi/Documents/wetten/csv/englandm-1718-E0.csv',sep=',', error_bad_lines=False, index_col=False)
#all_matfhes = pd.concat([all_matches,missing_part1], join='outer', axis=1).fillna(0)