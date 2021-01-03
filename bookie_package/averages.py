import pandas as pd
import sys
import numpy as np

def avg_goal_diff(df, new_column, a_h_team, a_h_goal_letter):
    """
    input: 
        df = dataframe with all results
        new_column = name of the new column
        a_h_team = HomeTeam or AwayTeam
        a_h_goal_letter = 'H' for home or 'A' for away
    output: 
        avg_per_team = dictionary with with team as key and columns as values with new column H/ATGDiff
    """
    df[new_column] = 0
    avg_per_team = {}
    all_teams = df[a_h_team].unique()
    for t in all_teams:
        df_team = df[df[a_h_team]==t].fillna(0)
        result = df_team['{}TGDiff'.format(a_h_goal_letter)].rolling(5).mean() 
        #print(t)
        #print(result)
        df_team[new_column] = result.shift(-5)
        #print(df_team[avg_h_a_diff])
        #7/0
        avg_per_team[t] = df_team
    return avg_per_team

def create_avg(df,columns,limits):
    homeAvgDF = create_avgPart(df,columns,'HomeTeam',limits)
    return create_avgPart(homeAvgDF,columns,'AwayTeam',limits)

def create_avgPart(df,columns, teamside,limits):
    """
    input: 
        df = dataframe with all results
        columns = columns to be selected for average value
        teamside = HomeTeam or AwayTeam or both as a list
        limits = a list which consists of number of rows to be averaged 
        --##odds_margin = odds margin down and up, as a condition
     output: 
        avg_per_team = dictionary with with team as key and columns as values with new column H/ATGDiff
    """
    df.sort_values(by='Date',ascending=True,inplace=True)
    
    #df[new_column] = 0
    avg_per_team = {}
    all_teams = df[teamside].unique()
    
    columns = [teamside[0:1]+col  for col in columns]
    
    #for leage in df['Div'].unique():
    #    df_leage = df[df['Div']==leage]
    
    for col in columns:
        for lim in limits:
            new_column = 'AvgLeage{}_{}'.format(col,lim)
            print(new_column,' to be calculated for ') 
           
            df.sort_values(['Div','Date'],inplace=True)
            df[col+'today'] =np.round(df.groupby(['Div','Date'])[col].transform(lambda x: x.mean()),4) 
            df_g = np.round(df.groupby(['Div','Date']).agg({col:'mean'}).groupby(level=0, group_keys=False).shift().rolling(lim).mean().reset_index(),4)
            df_g.rename(columns={col:new_column},inplace=True)
            df = df.merge(df_g,on=['Div','Date'], how='inner')
            
               
    #print(df)
    #df.to_excel('after.xlsx')
    print(df.describe())
    #7/0
    #df.replace(np.inf, np.finfo('float32').max,inplace=True)
    #print(df)
    #df.to_excel('after.xlsx')
  
    for t in all_teams:
        df_team = df[df[teamside]==t]
        df_team.fillna(df_team.mean(),inplace=True)
        for col in columns:
            for lim in limits:
                print(col,lim,' to be calculated for ',t)
                df_team['Avg{}_{}'.format(col,lim)] = df_team[col].shift().rolling(lim).mean() #result #shift(-5)
                df_team['{}_{}_Ratio'.format(col,lim)] = df_team['Avg{}_{}'.format(col,lim)] / df['AvgLeage{}_{}'.format(col,lim)]

        avg_per_team[t] = df_team[df_team[teamside]==t]

        
    return from_dict_value_to_df(avg_per_team)

def create_avgOld(df,avg_column, h_or_a_team, new_column,n=5):
    """
    input: 
        df = dataframe with all results
        avg_column = column to be selected for average value
        new_column = name of the new column
        h_or_a_team = HomeTeam or AwayTeam or both as a list
        n = number of rows to be averaged 
        --##odds_margin = odds margin down and up, as a condition
     output: 
        avg_per_team = dictionary with with team as key and columns as values with new column H/ATGDiff
    """
    df.sort_values(by='Date',ascending=True,inplace=True)
    df[new_column] = 0
    avg_per_team = {}
    all_teams = df[h_or_a_team[0]].unique()
    
    leageaAvgColumn = 'AvgLeage{}_{}'.format(new_column,n) 
    df[leageaAvgColumn] = df.apply(lambda x: df[(df['Div']==x['Div']) & \
                                    (df['season']==x['season'])].mean()[avg_column],axis=1)
    
    for t in all_teams:
        df_team = df[df[h_or_a_team[0]]==t].fillna(0)
        if len(h_or_a_team)==2:
            df_team = df[(df[h_or_a_team[0]]==t) | (df[h_or_a_team[1]]==t)].fillna(0)
        #oddsCondition = df['Avg{}'.format(h_or_team[0][0:1]) ] if 'GivenOdds' 
        #print(df_team)
        #print(t)
        #7/0
        #print(df_team.columns)
        
        
        df_team[new_column] = df_team[avg_column].shift().rolling(n).mean() #result #shift(-5)
        df_team[new_column+'Ratio'] = df_team[new_column] / df[leageaAvgColumn]

        avg_per_team[t] = df_team[df_team[h_or_a_team[0]]==t]

        #print(avg_per_team[t])
        #7/0
    return from_dict_value_to_df(avg_per_team)

def from_dict_value_to_df(d):
    """
    input = dictionary 
    output = dataframe as part of all the values from the dictionary
    """
    df = pd.DataFrame()
    for v in d.values():
        df = df.append(v)
    df.sort_values(by='Date',ascending=False,inplace=True)
    return df

def previous_data(df, h_or_a_team, column):
    """
    input: 
        df = dataframe with all results
        a_h_team = HomeTeam or AwayTeam
        column = column selected to get previous data from
    output:
        team_with_past_dict = dictionary with team as a key and columns as values with new 
                              columns with past value
    """
    d = dict()
    team_with_past_dict = dict()
    all_teams = df[h_or_a_team].unique()
    for team in all_teams:
        n_games = len(df[df[h_or_a_team]==team])
        team_with_past_dict[team] = df[df[h_or_a_team]==team]
        #print(team_with_past_dict[team].columns)
        for i in range(1, (6 if n_games>6 else n_games)):
            d[i] = team_with_past_dict[team].assign(
                result=team_with_past_dict[team].groupby(h_or_a_team)[column].shift(-i)

            ).fillna({'{}_X'.format(column): 0})
            a_h_goal_letter_inv = 'H' if h_or_a_team[0:1] == 'A' else 'A'
            team_win_odd = team_with_past_dict[team]['Avg{}'.format(h_or_a_team[0:1])]
            odds_reliability = team_win_odd / (team_win_odd + team_with_past_dict[team]['Avg{}'.format(a_h_goal_letter_inv)])

            team_with_past_dict[team]['{}_{}'.format(column, i)] = d[i].result * odds_reliability
    return team_with_past_dict


  