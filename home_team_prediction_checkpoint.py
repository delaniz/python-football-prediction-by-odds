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
league_mask = all_matches['Div'].str.contains('E|D|S|F|I') #for english premierleague
latest_seasons = all_matches[ time_mask & league_mask]
#print(season_1819.head(10))
#print(season_1819.tail(10))
#7/0
latest_seasons.fillna(0)
latest_seasons.set_index = 'Date'
latest_seasons.rename(columns={'HTGDiff':'HGoalDiff','ATGDiff':'AGoalDiff'},inplace=True)
latest_seasons = latest_seasons[['Date','HomeTeam','AwayTeam','FTHG','FTAG','AvgA','AvgD','AvgH','HGoalDiff','AGoalDiff','HST','AST']] #'Avg>2.5','Avg<2.5'
latest_seasons.sort_values(['Date'], ascending=True,inplace=True)

# calculates goal weight and value according to odds
#HTGW = HomeTeam Goal Weight
#HTTGV = HomeTeam Total Goal Value
latest_seasons['HGoalWeight'] = latest_seasons['AvgH'] / (latest_seasons['AvgH'] + latest_seasons['AvgA']+latest_seasons['AvgD'])
latest_seasons['HGoalWeighted'] = latest_seasons['HGoalWeight']*latest_seasons['FTHG']
latest_seasons['HGoalDiffWeighted'] = latest_seasons['HGoalWeight']*latest_seasons['HGoalDiff']
latest_seasons['AGoalWeight'] = latest_seasons['AvgA'] / (latest_seasons['AvgH'] + latest_seasons['AvgA']+latest_seasons['AvgD'])
latest_seasons['AGoalWeighted'] = latest_seasons['AGoalWeight']*latest_seasons['FTAG']
latest_seasons['AGoalDiffWeighted'] = latest_seasons['AGoalWeight']*latest_seasons['AGoalDiff']

#calculates the average home team total goal value according to given odds # HTTGVal,AvgHTTGVal
d_both_seasons = bp.averages.create_avg(latest_seasons, 'HTTGVal', 'HomeTeam', 'AvgHTTGVal')
#print(d_both_seasons)
df_both_seasons = bp.averages.from_dict_value_to_df(d_both_seasons)

#df_both_seasons=df_both_seasons.sort_values(['Year', 'Month','Day'], ascending=False)

# calculates the average home team goal difference across the last 10 hosting games
d_both_seasons = bp.averages.create_avg(df_both_seasons, 'HTGDiffVal', 'HomeTeam', 'AvgHTGDiffVal')

df_both_seasons = bp.averages.from_dict_value_to_df(d_both_seasons)


# calculates the average goals shot by the home team across the last 5 hosting games
avg_fthg_per_team=bp.averages.create_avg(df_both_seasons, 'FTHG', 'HomeTeam', 'AvgFTHG')

df_both_seasons = bp.averages.from_dict_value_to_df(avg_fthg_per_team)

# calculates the average home team goal difference across the last 5 hosting games
d_both_seasons = bp.averages.create_avg(df_both_seasons, 'HTGDiff', 'HomeTeam', 'AvgHTGDiff')

df_both_seasons = bp.averages.from_dict_value_to_df(d_both_seasons)

# calculates the average home team shots on target across the last 5 hosting games
d_both_seasons = bp.averages.create_avg(df_both_seasons, 'HST', 'HomeTeam', 'AvgHST')

df_both_seasons = bp.averages.from_dict_value_to_df(d_both_seasons)

#df_both_seasons=df_both_seasons.sort_values(['Year', 'Month','Day'], ascending=False)

#df_both_seasons=df_both_seasons.sort_values(['Year', 'Month','Day'], ascending=False)

print(df_both_seasons.columns)

#print(df_both_seasons.head(50))
#print(df_both_seasons.tail(10))
#print(df_both_seasons.describe())
#7/0
#print(df_both_seasons[df_both_seasons['HomeTeam']=='Arsenal'].head(10).)
#1/0

"""## Add Columns with previous HTGDiff and HST for each HomeTeam"""
"""
# HTGDiff values from the last ten home team games, per past match
team_with_past_HTGDiff=bp.averages.previous_data(df_both_seasons, 'HomeTeam', 'HTGDiff')

df_team_with_past_HTGDiff = bp.averages.from_dict_value_to_df(team_with_past_HTGDiff)
#print(df_team_with_past_HTGDiff.head(10))
#print(df_team_with_past_HTGDiff.tail(10))
#7/0
"""
"""  2.5+ goals percentage """
#print(df_team_with_past_HTGDiff.head(5)['HomeTeam'])
#df_team_with_past_HTGDiff[new_column] = df_team_with_past_HTGDiff.apply(lambda x:  float(len(df_team_with_past_HTGDiff.head(10)[df_team_with_past_HTGDiff[h_or_a_team]==x[h_or_a_team] & ((int(df['FTAG'])+int(df['FTHG']))>over_limit)])) / 10)

#df_team_with_past_HTOverPer = bp.averages.over_percentage(df_team_with_past_HTGDiff, 'HomeTeam', 'HTOverPer')
#df_team_with_past_HTGDiff['Total_Goal'] = df_team_with_past_HTGDiff['FTAG'] + df_team_with_past_HTGDiff['FTHG']
#t = df_team_with_past_HTGDiff.copy()
#df_team_with_past_HTGDiff['HTOverPer'] = t.apply(lambda x: float(len(t[t['HomeTeam']==x['HomeTeam']].head(10)[t['Total_Goal']>2.5]))/10, axis=1)

""" TODO over percentage  """
#print(df_team_with_past_HTGDiff.head(2).to_string())
#7/0
#Avg>2.5','Avg<2.5'
columns_HTGDiff = [  'Day', 'Month', 'Year', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG','AvgH','AvgD', 'AvgA',
    'HTGDiff', 'ATGDiff', 'AvgFTHG','AvgHTDiff','AvgHTTGVal','AvgHTGDiffVal','HST', 'AST',
      'HTGDiff_1', 'HTGDiff_2', 'HTGDiff_3', 'HTGDiff_4', 'HTGDiff_5']
    #'HTGDiff_6', 'HTGDiff_7',
    #'HTGDiff_8', 'HTGDiff_9', 'HTGDiff_10','' ]

#df_team_with_past_HTGDiff = df_both_seasons.reindex(columns=columns_HTGDiff)

#df_team_with_past_HTGDiff.fillna(0, inplace=True)

# HST values from the last ten home team games, per past match
"""
print(df_team_with_past_HTGDiff)
team_with_past_HST=bp.averages.previous_data(df_team_with_past_HTGDiff, 'HomeTeam', 'HST')

df_team_with_past_HST = bp.averages.from_dict_value_to_df(team_with_past_HST)

columns_HST =  ['HST_1', 'HST_2', 'HST_3', 'HST_4', 'HST_5'] # 'HST_6', 'HST_7', 'HST_8', 'HST_9', 'HST_10']
columns_HTGDiff_HST = columns_HTGDiff + columns_HST

df_team_with_past_HST = df_team_with_past_HST.reindex(columns=columns_HTGDiff_HST)

df_team_with_past_HST.fillna(0, inplace=True)
#df_team_with_past_HST.sort_values(['Year', 'Month','Day'], ascending=False,inplace=True)

# FTHG values from the last ten home team games, per past match
team_with_past_FTHG = bp.averages.previous_data(df_team_with_past_HST, 'HomeTeam', 'FTHG')

df_team_with_past_FTHG = bp.averages.from_dict_value_to_df(team_with_past_FTHG)

columns_FTHG = ['FTHG_1', 'FTHG_2', 'FTHG_3', 'FTHG_4', 'FTHG_5'] # 'FTHG_6', 'FTHG_7', 'FTHG_8', 'FTHG_9', 'FTHG_10']
columns_HTGDiff_HST_FTHG = columns_HTGDiff_HST + columns_FTHG

df_team_with_past_FTHG = df_team_with_past_FTHG.reindex(columns=columns_HTGDiff_HST_FTHG)
"""
      
#filling na to make data suitable for our model
df_both_seasons.fillna(0, inplace=True)

df_result = df_both_seasons.copy()
print(df_result.columns)
print(df_result.shape)
#7/0
df_result = df_result.drop(['HomeTeam', 'AwayTeam'], axis = 1)
df_both_seasons = df_both_seasons.drop(['HST'], axis = 1)

print(df_result.head())

print('Shape of features:', df_result.shape)

"""## Define Targets and drop Columns"""

# values I want to predict
target = df_result['FTHG']

# values we want to predict and are not necessary for the random forrest regressor 
# or were identified as noise
df_result= df_result.drop([
    'Date','FTHG','HTGDiff','HST','Month','Year','Day','HTTGVal','HTGDiffVal','HTGW'
], axis = 1) #'FTHG_4', 'FTHG_3','HST_5', 'FTHG_5'
print(df_result.columns)

"""## Training and Testing Sets"""

# splitting arrays into random train and test subsets
X_train, X_test, y_train, y_test = train_test_split(
    df_result, target, test_size = 0.25,random_state = 42
)

print('X_train Shape:', X_train.shape)
print('y_train Shape:', y_train.shape)
print('X_test Shape:', X_test.shape)
print('y_test Shape:', y_test.shape)

"""## Calculate Baseline"""

# as baseline we are going to use the HomeTeam Goal averages
features_names = list(df_result.columns)
X_train = np.array(X_train)
base = X_train[:, features_names.index('AvgFTHG')]
#subtracting train result from test data
baseline_errors = abs(base - y_train)
print('MAE: ', round(np.mean(baseline_errors), 2), 'Goals.')

"""## Train Model"""

# fitting the random forrest model at the begining with 1000 estimators
rf = bp.prediction.random_forrest(X_train, y_train, n_estimators=1000,random_state = 42)

"""## First Predictions on Test Data"""

bp.prediction.performance_accuracy(y_test,X_test, rf)

features=np.array(df_result)
predictions_FTHG = rf.predict(features)
next_games_predictions=np.round(predictions_FTHG,0)

#inserting predicted goals into original dataframe
df_both_seasons['FTHG'] = next_games_predictions
print(df_both_seasons.head())

"""### Single Decision Tree Visualising"""
"""
rf_depth_4 = bp.prediction.random_forrest(X_train, y_train, n_estimators=10,random_state = 42, max_depth = 4)

# randomly pick one tree from ten
tree_4 = rf_depth_4.estimators_[7]

# use export_graphviz to save the tree as a dot file first as indicated: 
# as described here: https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
export_graphviz(tree_4, out_file = 'tree_4.dot', feature_names = features_names, rounded = True, precision = 1)

# then use the dot file to create a png file 
(graph, ) = pydot.graph_from_dot_file('tree_4.dot')
graph.write_png('tree_4.png')

print('The depth of this tree is:', tree_4.tree_.max_depth)
"""
"""### Variable Importances in %"""

# creates a list of feature names and their importance
importance = np.round(rf.feature_importances_,4)
dictionary = dict(zip(features_names, importance))
sorted_dictionary=sorted(dictionary.items(), key=lambda x:x[1], reverse=True)
names=[]
values=[]
for i in range(0, len(importance)):
    print('Feature Importance: {:15} {}%'.format(
        sorted_dictionary[i][0], np.round(sorted_dictionary[i][1]*100,4))
         )
    names.append(sorted_dictionary[i][0])
    values.append(np.round(sorted_dictionary[i][1]*100,4))

"""## Feature Reduction"""
"""
sns.set(style='whitegrid', rc={'figure.figsize':(11.7,8.27)})
sns.set_context('talk')

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
bottom, top = ylim()
bottom = 0
cum_values=np.cumsum(values)
plt.plot(names,cum_values, '--bo', color='r')
# set importance at 95%
plt.axhline(95,color='black')
plt.xticks(rotation=90);
plt.xlabel('Feature'); 
plt.ylabel('Percentage'); 
plt.title('Cumulative Feature Importance');
plt.show()
"""
"""## Random Forest Optimization through Random Search"""

from sklearn.model_selection import RandomizedSearchCV

rs = bp.prediction.random_search(X_train,y_train,cv=10)

best_params = rs.best_params_

#best params calculated by our search algorithm (rs)
best_params

# reuses newly calculated params
rfc = bp.prediction.random_forrest(
    X_train, y_train, 
    n_estimators=best_params['n_estimators'],
    random_state = 42,
    min_samples_split = best_params['min_samples_split'],
    max_leaf_nodes = best_params['max_leaf_nodes'],
    max_features = best_params['max_features'],
    max_depth = best_params['max_depth'],
    bootstrap = best_params['bootstrap']
)

# recalculates new Mean Absolute Error and accuracy
bp.prediction.performance_accuracy(y_test,X_test, rfc)

next_games=df_result
# predicts new results with newly calculated params
#def randomPredict(trees_n):
#    for i in range(0,trees_n):
predictions_next_games = rf.predict(next_games)
next_games_predictions=np.round(predictions_next_games,0)
df_both_seasons['FTHG'] = next_games_predictions
print(df_both_seasons.head(20))
df_both_seasons.sort_values(by=['Date','HomeTeam','AwayTeam'],inplace=True,ascending=False)
df_both_seasons = df_both_seasons.drop(['AvgA'],axis=1)
df_both_seasons.to_excel('df_both_seasons_home.xlsx')

#randomPredict(10)
