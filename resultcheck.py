# Commented out IPython magic to ensure Python compatibility.
# %pylab inline
import pandas as pd
import numpy as np
import scipy 
import sys
import bookie_package as bp
import warnings
import dill
warnings.filterwarnings('ignore')

#load results
df = pd.read_excel('predictions_AvgRatioPoints.xlsx')
#load predictor
randomForestH = dill.load(open("randomForestH.pickle", "rb"))
randomForestA = dill.load(open("randomForestA.pickle", "rb"))

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.mean(),inplace=True)
df_nextGames = df[df['Date'] >= pd.to_datetime('today').date()]
#df_nextGames['HGoals'] = 3
#df_nextGames['AGoals'] = 2
df_nextGames_h =  df_nextGames[['HGoalExp_5',
                        'HGoalDiff_5_Ratio','HGoalDiff_3_Ratio',
                        'HShots_5_Ratio',
                        'HCorners_5_Ratio',
                        'AvgHGoalsWeighted_8',
                        
                        'HGoals_5_Ratio',
                        'AGoalsAllowed_5_Ratio',
                        'AGoalDiff_5_Ratio'
                        'AvgHomeGoalDiff_GivenOdds','AvgHomeGoalDiff_Head2Head']]
df_nextGames_a = df_nextGames[['AGoalExp_5',
                        'AGoalDiff_5_Ratio','AGoalDiff_3_Ratio',
                         'AShots_5_Ratio',        
                         'ACorners_5_Ratio',
                         'AvgAGoalsWeighted_8',
                       'HGoalDiff_5_Ratio',
                        'AGoals_5_Ratio',  
                        'HGoalsAllowed_5_Ratio',
                        'AvgAwayGoalDiff_GivenOdds', 'AvgAwayGoalDiff_Head2Head']]
print(df_nextGames_h)
df_nextGames['pre_HGoals'] = randomForestH.predict(df_nextGames_h)
df_nextGames['pre_AGoals'] = randomForestA.predict(df_nextGames_a)
df_nextGames.to_excel('nextGamesPredictions2.xlsx')

