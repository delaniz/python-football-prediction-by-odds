# Football-Score-Prediction-By-Odds-And-History

![github](https://img.shields.io/github/workflow/status/edeng23/binance-trade-bot/binance-trade-bot)


> Predicts match score based on history of both teams and actual given odds by bookmaker

## Why?

This project was inspired by the observation that the match history of teams gives strong hints how the next match is going to end by taking into account the given odds by bookmaker.
Because bookmakers has more powerfull tools and algorithms to calculate the right odds for them. We can use these odds to find out if there is a possibility to get advantage of it.


## How?

There are multiple attributes, which are averaged data of both teams in order to predict a score:
These are: 	
<table><tr><td>AvgGoalDiff_13</td><td> AvgGoalDiff_8</td><td> AvgGoalDiff_5</td><td> AvgGoalDiff_3</td></tr> 
		<tr><td>AvgShots_8</td><td> AvgShots_5</td><td> AvgShots_3</td><td></td></tr>
		<tr><td>AvgCorners_8</td><td>AvgCorners_5</td><td>AvgCorners_3</td><td></td></tr>
		<tr><td>AvgGoals_8</td><td>AvgGoals_5</td><td>AvgGoals_3</td><td></td></tr>
		<tr><td>AvgHomeGoalDiff_GivenOdds</td><td>AvgAwayGoalDiff_GivenOdds</td><td></td><td></td></tr>
		<tr><td>AvgHomeGoalDiff_Head2Head</td><td>AvgAwayGoalDiff_Head2Head</td><td></td><td></td></tr>
</table>

Average of the match history based on the fibonacci order like 3-5-8-13-21-35. The number at the end of the attribute names stands for amount of averaged games.
