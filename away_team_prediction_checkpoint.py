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

#all_matches = pd.read_pickle('all_data_2017_2021')
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

"""## Add avg Home Team Goal Difference"""
time_mask = (all_matches['Date']>='2018-07-01') & (all_matches['Date']<'2020-12-30')
league_mask = all_matches['Div'].str.contains('E|D|S|F|I') #for english premierleague
latest_seasons = all_matches[ time_mask & league_mask]
#print(latest_seasons.head(10))
#print(latest_seasons.tail(10))
#7/0
latest_seasons.fillna(0)
latest_seasons.set_index = 'Date'
latest_seasons = latest_seasons[['Date','Year','Month','Day','HomeTeam','AwayTeam','FTAG','AvgA','AvgD','AvgH','ATGDiff','AST']]
"""## Add avg Away Team Goal Difference"""
latest_seasons.sort_values(['Year', 'Month','Day'], ascending=True,inplace=True)
#print(latest_seasons.head())
#print(latest_seasons.tail())


# calculates goal weight and value according to odds
#HTGW = HomeTeam Goal Weight
#HTTGV = HomeTeam Total Goal Value
latest_seasons['ATGW'] = latest_seasons['AvgA'] / (latest_seasons['AvgH'] + latest_seasons['AvgA']+latest_seasons['AvgD'])
latest_seasons['ATTGVal'] = latest_seasons['ATGW']*latest_seasons['FTAG']
latest_seasons['ATGDiffVal'] = latest_seasons['ATGW']*latest_seasons['ATGDiff']

#calculates the average home team total goal value according to given odds # HTTGVal,AvgHTTGVal
d_both_seasons = bp.averages.create_avg(latest_seasons, 'ATTGVal', 'AwayTeam', 'AvgATTGVal')
#print(d_both_seasons)
df_both_seasons = bp.averages.from_dict_value_to_df(d_both_seasons)

#df_both_seasons=df_both_seasons.sort_values(['Year', 'Month','Day'], ascending=False)

# calculates the average home team goal difference across the last 10 hosting games
d_both_seasons = bp.averages.create_avg(df_both_seasons, 'ATGDiffVal', 'AwayTeam', 'AvgATGDiffVal')

df_both_seasons = bp.averages.from_dict_value_to_df(d_both_seasons)

# calculates the average away team goal difference across the last 10 hosting games
d_both_seasons = bp.averages.create_avg(df_both_seasons, 'ATGDiff', 'AwayTeam', 'AvgATGDiff')

df_both_seasons = bp.averages.from_dict_value_to_df(d_both_seasons)

#df_both_seasons=df_both_seasons.sort_values(['Year', 'Month','Day'], ascending=False)

# calculates the average goals shot by the home team across the last 10 hosting games
avg_ftag_per_team=bp.averages.create_avg(df_both_seasons, 'FTAG', 'AwayTeam', 'AvgFTAG')

df_both_seasons = bp.averages.from_dict_value_to_df(avg_ftag_per_team)

# calculates the average away team shots on target across the last 5 hosting games
d_both_seasons = bp.averages.create_avg(df_both_seasons, 'AST', 'AwayTeam', 'AvgAST')

df_both_seasons = bp.averages.from_dict_value_to_df(d_both_seasons)
#df_both_seasons=df_both_seasons.sort_values(['Year', 'Month','Day'], ascending=False)

"""## Add Columns with previous AST for each AwayTeam"""
"""
# HTGDiff values from the last ten home team games, per past match
team_with_past_ATGDiff=bp.averages.previous_data(df_both_seasons, 'AwayTeam', 'ATGDiff')

df_team_with_past_ATGDiff = bp.averages.from_dict_value_to_df(team_with_past_ATGDiff)

# AST values from the last ten home team games, per past match
team_with_past_AST=bp.averages.previous_data(df_team_with_past_ATGDiff, 'AwayTeam', 'AST')

df_team_with_past_AST = bp.averages.from_dict_value_to_df(team_with_past_AST)
columns_AST = [
    'Day', 'Month', 'Year', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTGDiff', 'ATGDiff', 'AvgFTAG','AvgATGDiff','AvgATTGVal','AvgATGDiffVal',
    'HST', 'AST','ATGDiff_1', 'ATGDiff_2', 'ATGDiff_3', 'ATGDiff_4', 'ATGDiff_5', 'AST_1', 'AST_2', 'AST_3', 'AST_4', 'AST_5','AvgA','AvgD','AvgH'] # 'AST_6', 'AST_7', 'AST_8', 'AST_9', 'AST_10',
    
#Avg>2.5','Avg<2.5'
df_team_with_past_AST = df_team_with_past_AST.reindex(columns=columns_AST)

df_team_with_past_AST.sort_values(['Year', 'Month','Day'], ascending=False,inplace=True)

df_team_with_past_AST.fillna(0, inplace=True)

# FTAG values from the last ten home team games, per past match
team_with_past_FTAG = bp.averages.previous_data(df_team_with_past_AST, 'AwayTeam', 'FTAG')

df_team_with_past_FTAG = bp.averages.from_dict_value_to_df(team_with_past_FTAG)

columns_FTAG = ['FTAG_1', 'FTAG_2', 'FTAG_3', 'FTAG_4', 'FTAG_5'] # 'FTAG_6', 'FTAG_7', 'FTAG_8', 'FTAG_9', 'FTAG_10'] 
columns_AST_FTHG = columns_AST + columns_FTAG

df_team_with_past_FTAG = df_team_with_past_FTAG.reindex(columns=columns_AST_FTHG)

df_team_with_past_FTAG.sort_values(['Year', 'Month','Day'], ascending=False,inplace=True)
"""
df_both_seasons.fillna(0, inplace=True)

df_both_seasons.columns

df_result = df_both_seasons.copy()

df_result = df_result.drop(['HomeTeam', 'AwayTeam'], axis = 1)

print('Shape of features:', df_result.shape)

"""## Features and Labels and Convert Data to Arrays"""

# values I want to predict
target = df_result['FTAG']

# values we want to predict and are not necessary for the random forrest regressor 
# or were identified as noise
df_result= df_result.drop([
    'Date','Day', 'Month','Day','Year','FTAG', 'ATGDiff', 'AST','ATTGVal','ATGDiffVal','ATGW'
], axis = 1)
#'AST_7', 'AST_8','AST_6',  'AST_10', 'AST_9', AST_5

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
base = X_train[:, features_names.index('AvgFTAG')]
# subtracting train result from test data
baseline_errors = abs(base - y_train)
print('MAE: ', round(np.mean(baseline_errors), 2), 'Goals.')

"""## Train Model"""

# fitting the random forrest model at the begining with 1000 estimators
rf = bp.prediction.random_forrest(X_train, y_train, n_estimators=1000,random_state = 42)

"""## First Predictions on Test Data"""

bp.prediction.performance_accuracy(y_test,X_test, rf)

next_games=df_result
predictions_next_games = rf.predict(next_games)
next_games_predictions=np.round(predictions_next_games,0)

del df_both_seasons['AST']

df_both_seasons['FTAG'] = next_games_predictions
print(df_both_seasons.head())

"""### Single Decision Tree Visualizing"""

rf_depth_4 = bp.prediction.random_forrest(X_train, y_train, n_estimators=10,random_state = 42, max_depth = 4)

# randomly pick one tree from ten
tree_4 = rf_depth_4.estimators_[7]

# use export_graphviz to save the tree as a dot file first as indicated: 
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
export_graphviz(tree_4, out_file = 'tree_4.dot', feature_names = features_names, rounded = True, precision = 1)

# then use the dot file to create a png file 
#(graph, ) = pydot.graph_from_dot_file('tree_4.dot')
#graph.write_png('tree_4_away.png');

#print('The depth of this tree is:', tree_4.tree_.max_depth)

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

"""## Random Forest Optimization through Random Search"""

from sklearn.model_selection import RandomizedSearchCV
rs = bp.prediction.random_search(X_train,y_train, cv=5)

best_params = rs.best_params_

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
predictions_next_games = rf.predict(next_games)
next_games_predictions=np.round(predictions_next_games,0)

df_both_seasons['FTAG'] = next_games_predictions
print(df_both_seasons.head(60))
df_both_seasons.sort_values(by=['Date','HomeTeam','AwayTeam'],inplace=True,ascending=False)
df_both_seasons = df_both_seasons.drop(['AvgH','AvgD'],axis=1)
df_both_seasons.to_excel('df_both_seasons_away.xlsx')