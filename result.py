# Commented out IPython magic to ensure Python compatibility.
# %pylab inline
import pandas as pd
import numpy as np
import scipy 
import sys
import bookie_package as bp
import warnings
warnings.filterwarnings('ignore')

# load results from 'home_team_prediction.ipynb' and 'away_team_prediction.ipynb'
df_home = pd.read_excel('df_both_seasons_home.xlsx')
df_away = pd.read_excel('df_both_seasons_away.xlsx')

cols_to_use = df_home.columns.difference(df_away.columns)
print(df_home.shape)
print(df_away.shape)
#print(df_home[cols_to_use].head(100).to_string())
#7/0
df_both = pd.merge(df_away, df_home[cols_to_use], left_index=True, right_index=True, how='outer')

del df_both['Unnamed: 0']
#print(df_both.head(100).to_string())
#print(df_both.tail(20))
#print(df_both[df_both['AwayTeam']=='Napoli'])
#print(df_home[df_home['AwayTeam']=='Napoli'])
#7/0
# create predicted goal differences subtracting predicted home and away goals from each other and vice vers
df_both['p_HTGDiff'] = df_both['FTHG'] - df_both['FTAG']
df_both['p_ATGDiff'] = df_both['FTAG'] - df_both['FTHG']

df_both.rename(columns={"HTGDiff": "r_HTGDiff", "ATGDiff": "r_ATGDiff", 'FTHG': 'p_HG', 'FTAG':'p_AG'}, inplace=True)

df_both = df_both.reindex(columns = ['Date', 'HomeTeam', 'AwayTeam', 'p_HG', 'p_AG',
                                   'p_HTGDiff', 'p_ATGDiff','r_HTGDiff', 'r_ATGDiff', 
                                   'AvgHTGDiff','AvgATGDiff','AvgHTGDiffVal','AvgATGDiffVal','AvgHTTGVal','AvgATTGVal',
                                   'AvgFTHG','AvgFTAG','AvgH','AvgD','AvgA']) #*r_HTGDiff, test_ATGDiff, AvgATGDiff

df_both.to_excel('both.xlsx')
df_both.fillna(0)
df_both.sort_values(by=['Date'], ascending=False)
print(df_both.head(50))
#print(df_both.tail(5))
# counting where error = 0 which means prediction and test data are the same = success
# then dividing it by the length of all errors
errors = abs(df_both['p_HTGDiff'] - df_both['r_HTGDiff'])
accuracy = (errors==0).sum() / len(errors) * 100
print('MAE:', round(np.mean(errors),2), 'Goals.')
print('Accuracy of Score Prediction:', round(accuracy, 2), '%.')

home_win_rate = len(df_both[(df_both['p_HTGDiff']> df_both['r_HTGDiff']) & (df_both['r_HTGDiff']>0)] ) / len(df_both)*100
#accuracy = (errors>=0).sum() / len(errors) * 100
#print('MAE:', round(np.mean(errors),2), 'Goals.')
print('Accuracy of Home Win Prediction with wrong Result:', round(home_win_rate, 2), '%.')

away_win_rate = len(df_both[(df_both['p_HTGDiff']< df_both['r_HTGDiff']) & (df_both['r_HTGDiff']<0)] ) / len(df_both)*100
#print('MAE:', round(np.mean(errors),2), 'Goals.')
print('Accuracy of Away Win Prediction with wrong Result:', round(away_win_rate, 2), '%.')
print('Accuracy of Win/Lost Prediction Total: {} %'.format(np.round(((accuracy+home_win_rate+away_win_rate),2))))
total_wins=(df_both["p_HTGDiff"] > 0).sum()
total_draw=(df_both["p_HTGDiff"] == 0).sum()
total_loss=(df_both["p_HTGDiff"] < 0).sum()

common_win = ((df_both["r_HTGDiff"] > 0) & (df_both["p_HTGDiff"] > 0)).sum()
common_draw = ((df_both["r_HTGDiff"] == 0) & (df_both["p_HTGDiff"] == 0)).sum()
common_lost = ((df_both["r_HTGDiff"] < 0) & (df_both["p_HTGDiff"] < 0)).sum()

print('Correct Prediction Total: {} %'.format(np.round(((common_win+common_draw+common_lost)/df_both.shape[0]) * 100,2)))
print('Correct Prediction Share Wins: {} %'.format(np.round((common_win /total_wins)*100, 2)))
print('Correct Prediction Share Draws: {} %'.format(np.round((common_draw / total_draw)*100,2)))
print('Correct Prediction Share Lost: {} %'.format(np.round((common_lost / total_loss)*100,2)))

#df_both = df_both.drop(['Month','Day','Year'],axis=1)
#df_both_filtered = df_both[(df_both['Date']>'2020-10-23') & (df_both['AvgH']>1.55) & (df_both['AvgH']<1.75)].head(60)
#df_both = df_both.drop(['Date'],axis=1)
#print(df_both_filtered)
#df_both['avgHTGDiff_now'] = df_both['r_HTGDiff']*(df_both['AvgH']/(df_both['AvgH']+df_both['AvgA']))
#print(df_both['avgHTGDiff_now'].rolling(window=5).mean()) #['avgHTGDiff_now']
#print(df_both.groupby('HomeTeam')['avgHTGDiff_now'].rolling(5).mean())
#7/0
#df_both['avgHTGDiff_5'] = df_both.groupby('HomeTeam')['avgHTGDiff_now'].rolling(5).mean()
#print(df_both)
#df_both.fillna(0)
#df_both['avgHTGDiff_5'] = df_both.apply(lambda x: x[['avgHTGDiff_now']].rolling(window=5).mean()['avgHTGDiff_now'],axis=1)
#df_both.groupby('HomeTeam').transform(lambda x: pd.rolling(x['avgHTGDiff_now'], window=5).mean(),axis=1)
#print(df_both[df_both['HomeTeam']=='Atalanta'].head(10))
#print(df_both[df_both['HomeTeam']=='Atalanta']['avgHTGDiff_now'].head(6).rolling(window=5).mean())
#print(df_both[df_both['AwayTeam']=='Sampdoria']['avgHTGDiff_now'].head(6).rolling(window=5).mean())