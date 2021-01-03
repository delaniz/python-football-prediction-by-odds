# Commented out IPython magic to ensure Python compatibility.
# %pylab inline
from matplotlib.pyplot import ylim
import pandas as pd
import seaborn as sns
import numpy as np
import scipy 
import matplotlib.pyplot as plt
import matplotlib.style as style
from sklearn.ensemble import RandomForestRegressor
import sys
from sklearn.model_selection import train_test_split
import pydot
#imports the own created package
import bookie_package as bp
from sklearn.tree import export_graphviz
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

def try_parsing_date(text):
    for fmt in ('%d/%m/%Y','%d/%m/%y','%Y-%m-%d', '%d.%m.%Y'):
        try:
            datetime.strptime(text, fmt)
            return fmt
        except ValueError:
            pass
    raise ValueError('no valid date format found')

#imports pickle file created and saved in 'load_and_clean.ipynb'
all_matches = pd.read_csv('all_data_2017_2021.csv',sep=',', error_bad_lines=False, index_col=False,engine='python',
                            parse_dates=['Date'],date_parser=lambda x: pd.to_datetime(x, format=try_parsing_date(x)))
#pd.read_pickle('all_data_2017_2021')
all_matches.sort_values(by='Date',inplace=True)
#print(all_matches.head(10))
#print(all_matches.tail(10))
#print(all_matches[['HomeTeam','AwayTeam','AvgA','FTR','Month']].tail(10))
#7/0
"""## Add avg Home Team Goal Difference"""
time_mask = (all_matches['Date']>='2018-07-01') & (all_matches['Date']<'2020-12-30')
#league_mask = all_matches['Div'].str.contains() #for english premierleague
latest_seasons = all_matches[ time_mask] # & league_mask]
#print(season_1819.head(10))
#print(season_1819.tail(10))
#7/0
latest_seasons.fillna(latest_seasons.mean(),inplace=True)
latest_seasons.set_index = 'Date'
latest_seasons.rename(columns={'HTGDiff':'HGoalDiff','ATGDiff':'AGoalDiff','FTHG':'HGoals','FTAG':'AGoals' \
                               ,'HST':'HShots','AST':'AShots','HC':'HCorners','AC':'ACorners'},inplace=True)
latest_seasons['pre_HGoals'] = 0
latest_seasons['pre_AGoals'] = 0
latest_seasons = latest_seasons[['Div','Date','HomeTeam','AwayTeam','FTR','pre_HGoals','pre_AGoals','HGoals','AGoals',\
                        'PSH','PSD','PSA','HGoalDiff','AGoalDiff','HShots','AShots','HCorners','ACorners']] #'Avg>2.5','Avg<2.5'
latest_seasons.rename(columns={'PSH':'AvgH','PSD':'AvgD','PSA':'AvgA'},inplace=True)
#latest_seasons = latest_seasons[latest_seasons.duplicated(['Date', 'HomeTeam', 'AwayTeam'], keep=False)]
#print(latest_seasons)
#7/0
latest_seasons.sort_values(['HGoals','AGoals'], ascending=False).drop_duplicates(['Date','HomeTeam','AwayTeam'],inplace=True)

latest_seasons.sort_values(['Date'], ascending=True,inplace=True)
latest_seasons['season'] = latest_seasons['Date'].apply(lambda x: str(x.year)+"/"+str(x.year+1) if (x.month >= 7) else str(x.year-1)+"/"+str(x.year))

# calculates goal weight and value according to odds

latest_seasons['odds_sum']  = latest_seasons['AvgH']+latest_seasons['AvgD']+latest_seasons['AvgA']
latest_seasons['HGoalWeight'] = (latest_seasons['AvgH']*(latest_seasons['odds_sum']/2.5)*1.45) / latest_seasons['odds_sum']
latest_seasons['AGoalWeight'] = (latest_seasons['AvgA']*(latest_seasons['odds_sum']/2.5)*1.45) / latest_seasons['odds_sum']
latest_seasons['HGoalsWeighted'] = latest_seasons['HGoalWeight']*latest_seasons['HGoals']
latest_seasons['AGoalsWeighted'] = latest_seasons['AGoalWeight']*latest_seasons['AGoals']

latest_seasons['HGoalDiffWeighted'] = latest_seasons['HGoalWeight']*latest_seasons['HGoalDiff']
latest_seasons['AGoalDiffWeighted'] = latest_seasons['AGoalWeight']*latest_seasons['AGoalDiff']

#calculates the average home/away team total goal value according to given odds 
#df_both_seasons = bp.averages.create_avg(latest_seasons, 'HGoalsWeighted', 'HomeTeam', 'AvgHGoalsWeighted')
#df_both_seasons = bp.averages.create_avg(df_both_seasons, 'AGoalsWeighted', 'AwayTeam', 'AvgAGoalsWeighted')

# calculates the average home/away team goal difference across the last 5 hosting games (weighted by odds)
#df_both_seasons = bp.averages.create_avg(df_both_seasons, 'HGoalDiffWeighted', 'HomeTeam', 'AvgHGoalDiffWeighted')
#df_both_seasons = bp.averages.create_avg(df_both_seasons, 'AGoalDiffWeighted', 'AwayTeam',. 'AvgAGoalDiffWeighted')

#df_both_seasons = latest_seasons

latest_seasons['HGoalsAllowed'] = latest_seasons['AGoals']
latest_seasons['AGoalsAllowed'] = latest_seasons['HGoals'] 
latest_seasons['HPoints'] = np.where(latest_seasons['FTR']=='H', 3,np.where(latest_seasons['FTR']=='A',0,1))
latest_seasons['APoints'] = np.where(latest_seasons['FTR']=='H', 0,np.where(latest_seasons['FTR']=='A',3,1))

# calculates the average home/away team goal difference across the last 10 games (weighted by odds)
print(latest_seasons)

averageOfWeeks = [3,5,8,13] #average of 3 weeks, average of 5 weeks ...
average_columns = ['Goals','GoalsAllowed','GoalDiff','Points','Shots','Corners','GoalsWeighted','GoalDiffWeighted']
#latest_seasons['AvgLeageHGoals_todayTotal'] = 0
#for leage in latest_seasons['Div'].unique():
#    leageMaske = latest_seasons['Div']==leage
#    latest_seasons['AvgLeageHGoals_todayTotal'] = latest_seasons.apply(lambda x: x['AvgLeageHGoals_todayTotal'] if x['Div'] ['HGoals'].shift().rolling(3).mean()
#print(latest_seasons)
#7/0

df_both_seasons = bp.averages.create_avg(latest_seasons,average_columns,averageOfWeeks)
for week in averageOfWeeks:
    df_both_seasons['HGoalExp_{}'.format(week)] = df_both_seasons['HGoals_{}_Ratio'.format(week)]*df_both_seasons['AGoalsAllowed_{}_Ratio'.format(week)]*df_both_seasons['AvgLeageHGoals_{}'.format(week)]
    df_both_seasons['AGoalExp_{}'.format(week)] = df_both_seasons['AGoals_{}_Ratio'.format(week)]*df_both_seasons['HGoalsAllowed_{}_Ratio'.format(week)]*df_both_seasons['AvgLeageAGoals_{}'.format(week)]
    
# calculates the average home/away team given odds reliability
def getAvgDiff_GivenOdds(df,x,teamside):
    odd = 'Avg{}'.format(teamside[0:1])
    #print(x)
    #print(odd)
    margin_down = (x[odd]-0.08)
    margin_up =  (x[odd]+0.08)
    df_conditioned = df[(df[teamside]==x[teamside]) & (df[odd]>=margin_down) & (df[odd]<=margin_up) & (df['Date']<x['Date'])]
    return df_conditioned['{}GoalDiff'.format(teamside[0:1])].sum()/len(df_conditioned)

print(df_both_seasons)
df_both_seasons['AvgHomeGoalDiff_GivenOdds'] = df_both_seasons.apply(lambda x: getAvgDiff_GivenOdds(df_both_seasons,x,'HomeTeam'),axis=1)
print('home_givenodds finished')
df_both_seasons['AvgAwayGoalDiff_GivenOdds'] = df_both_seasons.apply(lambda x: getAvgDiff_GivenOdds(df_both_seasons,x,'AwayTeam'),axis=1)
print('away_givenodds finished')  
#df_both_seasons = bp.averages.create_avg(df_both_seasons, 'HGoalDiff', 'HomeTeam', 'AvgHomeGoalDiff_GivenOdds',5,0.5)
#df_both_seasons = bp.averages.create_avg(df_both_seasons, 'AGoalDiff', 'AwayTeam', 'AvgAwayGoalDiff_GivenOdds',5,0.5)

# calculates the average home/away team given odds reliability 
def getAvgDiff_Head2Head(df,x,teamside):
    teamside_inv = 'AwayTeam' if teamside == 'HomeTeam' else 'HomeTeam'
    #print(teamside + " vs. "+teamside_inv)
    #print(x[teamside])
    #print(x[teamside_inv])
    #print(x)
    df_conditioned = df[(df[teamside]==x[teamside]) & (df[teamside_inv]==x[teamside_inv]) & (df['Date']<x['Date'])]
    #print(df_conditioned)
    #7/0
    return df_conditioned['{}GoalDiff'.format(teamside[0:1])].sum()/len(df_conditioned)

df_both_seasons['AvgHomeGoalDiff_Head2Head'] = df_both_seasons.apply(lambda x: getAvgDiff_Head2Head(df_both_seasons,x,'HomeTeam'),axis=1)
print('home_head2head finished')
df_both_seasons['AvgAwayGoalDiff_Head2Head'] = df_both_seasons.apply(lambda x: getAvgDiff_Head2Head(df_both_seasons,x,'AwayTeam'),axis=1)

print('period data will be now averaged by given weigths')
averageOfWeeks = [3,5,8,13] #average of 3 weeks, average of 5 weeks ...
weigth = [3,2.2,1.8,1]
sides = ['H','A']
average_columns = ['Goals','GoalsAllowed','GoalDiff','Points','Shots','Corners','GoalsWeighted','GoalDiffWeighted']
for col in average_columns:
    for side in sides:
        df_both_seasons[side+col+'_Avg'] = (df_both_seasons['{}{}_{}_Ratio'.format(side,col,averageOfWeeks[0])]*weigth[0]+df_both_seasons['{}{}_{}_Ratio'.format(side,col,averageOfWeeks[1])]*weigth[1]+df_both_seasons['{}{}_{}_Ratio'.format(side,col,averageOfWeeks[2])]*weigth[2]+df_both_seasons['{}{}_{}_Ratio'.format(side,col,averageOfWeeks[3])]*weigth[3])/sum(weigth)
df_both_seasons['HGoalExp_Avg'] = (df_both_seasons['HGoalExp_{}'.format(averageOfWeeks[0])]*weigth[0]+df_both_seasons['HGoalExp_{}'.format(averageOfWeeks[1])]*weigth[1]+df_both_seasons['HGoalExp_{}'.format(averageOfWeeks[2])]*weigth[2]+df_both_seasons['HGoalExp_{}'.format(averageOfWeeks[3])]*weigth[3])/sum(weigth)
df_both_seasons['AGoalExp_Avg'] = (df_both_seasons['AGoalExp_{}'.format(averageOfWeeks[0])]*weigth[0]+df_both_seasons['AGoalExp_{}'.format(averageOfWeeks[1])]*weigth[1]+df_both_seasons['AGoalExp_{}'.format(averageOfWeeks[2])]*weigth[2]+df_both_seasons['AGoalExp_{}'.format(averageOfWeeks[3])]*weigth[3])/sum(weigth)
df_both_seasons['AvgHGoalsWeighted_Avg'] = (df_both_seasons['AvgHGoalsWeighted_{}'.format(averageOfWeeks[0])]*weigth[0]+df_both_seasons['AvgHGoalsWeighted_{}'.format(averageOfWeeks[1])]*weigth[1]+df_both_seasons['AvgHGoalsWeighted_{}'.format(averageOfWeeks[2])]*weigth[2]+df_both_seasons['AvgHGoalsWeighted_{}'.format(averageOfWeeks[3])]*weigth[3])/sum(weigth)
df_both_seasons['AvgAGoalsWeighted_Avg'] = (df_both_seasons['AvgAGoalsWeighted_{}'.format(averageOfWeeks[0])]*weigth[0]+df_both_seasons['AvgAGoalsWeighted_{}'.format(averageOfWeeks[1])]*weigth[1]+df_both_seasons['AvgAGoalsWeighted_{}'.format(averageOfWeeks[2])]*weigth[2]+df_both_seasons['AvgAGoalsWeighted_{}'.format(averageOfWeeks[3])]*weigth[3])/sum(weigth)


#df_both_seasons = df_both_seasons.drop(['HCorners','ACorners','HShots','AShots','AGoalDiff','HGoalDiff'],axis=1)
print(df_both_seasons)

#filling na to make data suitable for our model
df_both_seasons.fillna(df_both_seasons.mean(), inplace=True)
df_both_seasons.to_excel('avg_data.xlsx')
