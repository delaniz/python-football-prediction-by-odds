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
from sklearn import tree
#from sklearn.tree import export_graphviz     # this can be even > import export_graphviz <
import os
#import pydot
#import pickle

#from sklearn import tree
#from sklearn.tree import export_graphviz     # this can be even > import export_graphviz <
#from sklearn.externals. import StringIO   # shortened StringIO instead of six.StringIO

import warnings
warnings.filterwarnings('ignore')

df_both_seasons =  pd.read_excel('avg_data.xlsx')
del df_both_seasons['Unnamed: 0']

time_mask = (df_both_seasons['Date']>='2019-07-01') & (df_both_seasons['Date']<'2020-12-30')
#league_mask = all_matches['Div'].str.contains() #for english premierleague
df_both_seasons = df_both_seasons[ time_mask] # & league_mask]

"""## Training and Testing Sets"""

def predict_outcome(df_result,target,dot_name):
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
    """base = X_train[:, features_names.index(target_column)]
    #subtracting train result from test data
    baseline_errors = abs(base - y_train)
    print('MAE: ', round(np.mean(baseline_errors), 2), 'Goals.')
    """
    """## Train Model"""

    # fitting the random forrest model at the begining with 1000 estimators
    rf = bp.prediction.random_forrest(X_train, y_train, n_estimators=1000,random_state = 42)

    """## First Predictions on Test Data"""

    bp.prediction.performance_accuracy(y_test,X_test, rf)

    features=np.array(df_result)
    predictions = rf.predict(features)
    #pd.DataFrame(predictions).to_excel('predictions_raw'+dot_name+'.xlsx')
    from decimal import Decimal, ROUND_HALF_UP
    next_games_predictions= [int(Decimal(x).to_integral_value(rounding=ROUND_HALF_UP)) for x in predictions ]
    #pd.DataFrame(next_games_predictions).to_excel('predictions_rounded'+dot_name+'.xlsx')
    #np.round(predictions_FTHG,0)
    

    #inserting predicted goals into original dataframe
    #df_both_seasons['HGoals'] = next_games_predictions
    

    """### Single Decision Tree Visualising"""

    rf_depth_4 = bp.prediction.random_forrest(X_train, y_train, n_estimators=10,random_state = 42, max_depth = 4)

    # randomly pick one tree from ten
    tree_4 = rf_depth_4.estimators_[7]

    # use export_graphviz to save the tree as a dot file first as indicated: 
    # as described here: https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
    from io import StringIO
    import pydotplus
    #dot_data = StringIO()
    export_graphviz(tree_4, out_file = dot_name+".dot", feature_names = features_names, rounded = True, precision = 1)

   
    os.system('dot -Tpng '+dot_name+'.dot -o '+dot_name+'.png')
    #TODO later!
    #TODO
    print('The depth of this tree is:', tree_4.tree_.max_depth)

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

    """## Random Forest Optimization through Random Search"""

    from sklearn.model_selection import RandomizedSearchCV

    rs = bp.prediction.random_search(X_train,y_train,cv=10)

    best_params = rs.best_params_

    #best params calculated by our search algorithm (rs)
    print(best_params)
    """best_params = {'n_estimators': 434,
                    'min_samples_split': 10,
                    'max_leaf_nodes': 19,
                    'max_features': 0.6,
                    'max_depth': 5,
                    'bootstrap': False}
    """
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

    next_games=df_result #[df_result['Date']>=pd.to_datetime('today').date()]
    # predicts new results with newly calculated params
    #def randomPredict(trees_n):
    #    for i in range(0,trees_n):
    predictions_next_games = rf.predict(next_games)
    #from decimal import Decimal, ROUND_HALF_UP
    predictions_next_games_r= np.round(predictions_next_games,0) 
    #[int(Decimal(x).to_integral_value(rounding=ROUND_HALF_UP)) for x in predictions_next_games ]
    return [predictions_next_games,predictions_next_games_r,rf]

#df_both_seasons['HGoalExp'] = df_both_seasons['HGoals_5_Ratio']*df_both_seasons['AGoalsAllowed_5_Ratio']*df_both_seasons['AvgLeageHGoals_5']
#df_both_seasons['AGoalExp'] = df_both_seasons['AGoals_5_Ratio']*df_both_seasons['HGoalsAllowed_5_Ratio']*df_both_seasons['AvgLeageAGoals_5']
"""## Define Targets and drop Columns"""
#df_both_seasons['HGoalDiff']      = df_both_seasons['HGoals'] - df_both_seasons['AGoals']
#df_both_seasons['AGoalDiff']      = df_both_seasons['AGoals'] - df_both_seasons['HGoals']

df_result = df_both_seasons.copy()

df_result = df_result.drop(['HomeTeam', 'AwayTeam'], axis = 1)
#df_both_seasons = df_both_seasons.drop(['HST','AST'], axis = 1)

print(df_result.head())

#print('Shape of features:', df_result.shape)
df_result.replace([np.inf, -np.inf], np.nan, inplace=True)
df_result.fillna(df_result.mean(),inplace=True)
# values we want to predict and are not necessary for the random forrest regressor 
# or were identified as noise
#target_hgoals = df_result['HGoals']
#target_agoals = df_result['AGoals']

#df_result['AvgHomeGoalDiff_All'] = df_result['AvgHomeGoalDiff_General'] * df_result['AvgHomeGoalDiff']
#df_result['AvgAwayGoalDiff_All'] = df_result['AvgAwayGoalDiff_General'] * df_result['AvgAwayGoalDiff']
print(df_result.columns)

print(df_result)
#7/0

df_result_h =  df_result[['HGoalExp_Avg',
                        'HGoalDiff_Avg',
                        'HPoints_Avg',
                        'HGoals_Avg',
                        'HGoalsAllowed_Avg',
                       # 'HShots_Avg',
                       # 'HCorners_Avg',
                        'AvgHGoalsWeighted_Avg',
                        #'AGoalDiff_Avg',
                        #'HGoals_Avg',
                        #'AGoalsAllowed_Avg',
                        'AvgHomeGoalDiff_GivenOdds','AvgHomeGoalDiff_Head2Head']]
df_result_a = df_result[['AGoalExp_Avg',
                        'AGoalDiff_Avg',
                        'APoints_Avg',
                         'AGoals_Avg',
                        'AGoalsAllowed_Avg',
                        #'AShots_Avg',
                        #'ACorners_Avg',
                        'AvgAGoalsWeighted_Avg',
                       # 'HGoalDiff_Avg',
                       # 'AGoals_Avg',
                       # 'HGoalsAllowed_Avg',
                        'AvgAwayGoalDiff_GivenOdds', 'AvgAwayGoalDiff_Head2Head']]

df_result_diffHA = df_result_h-df_result_a.values
df_result_diffAH = df_result_a-df_result_h.values
df_result_diff = df_result_diffHA.merge(df_result_diffAH,on=df_result_diffHA.index,how='outer')
for col in ['HPoints_Avg','APoints_Avg',
                        'HGoalExp_Avg','AGoalExp_Avg',
                        'HGoalDiff_Avg','AGoalDiff_Avg',
                        'HGoals_Avg','AGoals_Avg',
                        'HGoalsAllowed_Avg', 'AGoalsAllowed_Avg',
                        'AvgHGoalsWeighted_Avg','AvgAGoalsWeighted_Avg',
                      'AvgHomeGoalDiff_GivenOdds','AvgAwayGoalDiff_GivenOdds',
                        'AvgHomeGoalDiff_Head2Head','AvgAwayGoalDiff_Head2Head']:
    df_both_seasons[col+'Diff'] = df_result_diff[col]
#print(df_result_diff)
#print(df_result_diff.columns)
#df_both_seasons.merge(df_result_diff,on=df_both_seasons.index,how='outer')
#print(df_both_seasons)
#print(df_both_seasons.columns)
#7/0
#print(df_result_diffHA)
#print(df_result_diffAH)
#print(df_result_diffHA.columns)
#print(df_result_diffAH.columns)
#7/0
algoName= 'predictions_DiffAB_AvgRatioNewByLastSeasonTrainingData'
#for lim in [3,5,8,13]:
import matplotlib.pyplot as plt
import seaborn as sns
def showPlot(x,y):
    # Making Pearson Thermograph
    # Mapping labels to 0 and 1
    #y_all=y_all.map({'NH':0,'H':1})
    # Merge feature sets and labels
    train_data=pd.concat([x,y],axis=1)
    colormap = plt.cm.RdBu
    plt.figure(figsize=(21,18))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(train_data.astype(float).corr(),linewidths=0.1,vmax=1.0,
                square=True, cmap=colormap, linecolor='white', annot=True)
    plt.show()
from sklearn import linear_model

def createGamePrediction(team1_vector, team2_vector, xTrain, yTrain):
    xTrain, X_test, yTrain, y_test = train_test_split(xTrain, yTrain)
    xTrain.shape, yTrain.shape
    X_test.shape, y_test.shape
    lm = linear_model.LinearRegression()
    model2 = lm.fit(xTrain, yTrain)
    diff = [a - b for a, b in zip(team1_vector, team2_vector)]
    predictions = lm.predict(diff)
    return predictions
    
showPlot(df_result_diffHA,df_result['HGoals'])
showPlot(df_result_diffAH,df_result['AGoals'])


X_trainH, X_testH, y_trainH, y_testH = train_test_split(
        df_result_h, df_result['HGoals'], test_size = 0.25,random_state = 42
    )
X_trainA, X_testA, y_trainA, y_testA = train_test_split(
        df_result_h, df_result['AGoals'], test_size = 0.25,random_state = 42
    )

resH = predict_outcome(df_result_diffHA,df_result['HGoals'],algoName+'_home')
resA = predict_outcome(df_result_diffAH,df_result['AGoals'],algoName+'_away')

df_both_seasons['pre_HGoals'] = resH[0]
df_both_seasons['pre_HGoals_dec'] = resH[1]
df_both_seasons['pre_AGoals'] = resA[0]
df_both_seasons['pre_AGoals_dec'] = resA[1]

randomForestH = resH[2]
randomForestA = resA[2]
import dill
# Save the file
dill.dump(randomForestH, file = open("randomForestH"+algoName+".pickle", "wb"))
dill.dump(randomForestA, file = open("randomForestA"+algoName+".pickle", "wb"))
# Reload the file
#company1_reloaded = dill.load(open("company1.pickle", "rb"))

df_both_seasons.sort_values(by=['Date','Div','HomeTeam'],inplace=True,ascending=False)
#df_both_seasons = df_both_seasons.drop(['AvgA'],axis=1)
prediction_columns = df_both_seasons[['Div','Date','HomeTeam','AwayTeam','pre_HGoals','pre_AGoals','HGoals','AGoals',
                                'AvgH','AvgA',
                                'HPoints_Avg','APoints_Avg',
                                'HGoalExp_Avg','AGoalExp_Avg',
                        'HGoalDiff_Avg','AGoalDiff_Avg',
                        'HGoals_Avg','AGoals_Avg',
                        'HGoalsAllowed_Avg', 'AGoalsAllowed_Avg',
                        #'HShots_Avg','AShots_Avg',
                        #'HCorners_Avg','ACorners_Avg',
                        'AvgHGoalsWeighted_Avg','AvgAGoalsWeighted_Avg',
                        #'AGoalDiff_Avg','HGoalDiff_Avg',
                        
                                
                   
                        'AvgHomeGoalDiff_GivenOdds','AvgAwayGoalDiff_GivenOdds',
                        'AvgHomeGoalDiff_Head2Head','AvgAwayGoalDiff_Head2Head']]
diff_columns = df_both_seasons[['Div','Date','HomeTeam','AwayTeam','pre_HGoals','pre_AGoals','HGoals','AGoals',
                                'AvgH','AvgA',
                                'HPoints_AvgDiff','APoints_AvgDiff',
                                'HGoalExp_AvgDiff','AGoalExp_AvgDiff',
                        'HGoalDiff_AvgDiff','AGoalDiff_AvgDiff',
                        'HGoals_AvgDiff','AGoals_AvgDiff',
                        'HGoalsAllowed_AvgDiff', 'AGoalsAllowed_AvgDiff',
                        #'HShots_Avg','AShots_Avg',
                        #'HCorners_Avg','ACorners_Avg',
                        'AvgHGoalsWeighted_AvgDiff','AvgAGoalsWeighted_AvgDiff',
                        #'AGoalDiff_Avg','HGoalDiff_Avg',
                        
                                
                   
                        'AvgHomeGoalDiff_GivenOddsDiff','AvgAwayGoalDiff_GivenOddsDiff',
                        'AvgHomeGoalDiff_Head2HeadDiff','AvgAwayGoalDiff_Head2HeadDiff']]
                       
prediction_columns.to_excel(algoName+'pColumns'+'.xlsx')
diff_columns.to_excel(algoName+'_DiffCols'+'.xlsx')
df_both_seasons.to_excel(algoName+'.xlsx')

def print_accuracy(column):
    preCol = 'pre_'+column+'_dec'
    naming = 'Home' if (column[0:1] == 'H') else 'Away'
    errors = abs(df_both_seasons[preCol] - df_both_seasons[column])
    accuracy = (errors==0).sum() / len(errors) * 100
    print('MAE:', round(np.mean(errors),2), 'Goals.')
    print('Accuracy of '+naming+' Score Prediction:', round(accuracy, 2), '%.')

print_accuracy('HGoals')
print_accuracy('AGoals')
df_both_seasons['pre_HGoalDiff'] = df_both_seasons["pre_HGoals_dec"]-df_both_seasons["pre_AGoals_dec"]
total_wins=(df_both_seasons["HGoalDiff"] > 0).sum()
total_draw=(df_both_seasons["HGoalDiff"] == 0).sum()
total_loss=(df_both_seasons["HGoalDiff"] < 0).sum()
common_win = ((df_both_seasons["HGoalDiff"] > 0) & (df_both_seasons["pre_HGoalDiff"] > 0)).sum()
common_draw = ((df_both_seasons["HGoalDiff"] == 0) & (df_both_seasons["pre_HGoalDiff"] == 0)).sum()
common_lost = ((df_both_seasons["HGoalDiff"] < 0) & (df_both_seasons["pre_HGoalDiff"] < 0)).sum()

print('Correct Prediction Total: {} %'.format(np.round(((common_win+common_draw+common_lost)/df_both_seasons.shape[0]) * 100,2)))
print('Correct Prediction Share Wins: {} %'.format(np.round((common_win /total_wins)*100, 2)))
print('Correct Prediction Share Draws: {} %'.format(np.round((common_draw / total_draw)*100,2)))
print('Correct Prediction Share Lost: {} %'.format(np.round((common_lost / total_loss)*100,2)))
#randomPredict(10)5