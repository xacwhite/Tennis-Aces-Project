## Tennis Aces Starting Project by Zach W.

#############
## Imports ##
#############

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


################################
## Load and Investigate Data: ##
################################

tennis = pd.read_csv("tennis_stats.csv")
print(tennis.head(30))
print(tennis.info())
print(tennis.describe())

get_player_info = tennis.columns[0:2]
get_variables = tennis.columns[2:19]
get_outcomes = tennis.columns[20:24]

player_info = []
variables = []
outcomes = []

for i in get_player_info:
    player_info.append(i)

for i in get_variables:
    variables.append(i)

for i in get_outcomes:
    outcomes.append(i)

print(player_info)
print(variables)
print(outcomes)


###########################
## Exploratory Analysis: ##
###########################

for var in variables:
    plt.subplots(2, 2, figsize = (10,10))

    ax = plt.subplot(2, 2, 1)
    ax.scatter(tennis[var], tennis["Wins"], alpha = 0.5)
    ax.set_xlabel(var)
    ax.set_ylabel("Wins")
    ax.set_title("Wins vs " + str(var)) 

    ax2 = plt.subplot(2, 2, 2)
    ax2.scatter(tennis[var], tennis["Losses"], alpha = 0.5)
    ax2.set_xlabel(var)
    ax2.set_ylabel("Losses")
    ax2.set_title("Losses vs " + str(var))
    
    ax3 = plt.subplot(2, 2, 3)
    ax3.scatter(tennis[var], tennis["Winnings"], alpha = 0.5)
    ax3.set_xlabel(var)
    ax3.set_ylabel("Winnings")
    ax3.set_title("Winnings vs " + str(var))

    ax4 = plt.subplot(2, 2, 4)
    ax4.scatter(tennis[var], tennis["Losses"], alpha = 0.5)
    ax4.set_xlabel(var)
    ax4.set_ylabel("Ranking")
    ax4.set_title("Ranking vs " + str(var))

    plt.show()

""" The BreakPointsFaced, BreakPointsOpportunities, DoubleFaults, ReturnGamesPlayed, and ServiceGamesPlayed 
 all had a strong positive correlation with all four outcomes, 'Wins', 'Losses', 'Winnings', 'Ranking'. """


#######################################
## Single Feature Linear Regressions ##
#######################################

lm = LinearRegression()

# finding and ploting the best scoring data
for var in variables:
    for outcome in outcomes:
        x_train, x_test, y_train, y_test = train_test_split(tennis[[var]], tennis[[outcome]], train_size = 0.8, test_size = 0.2)
        lm.fit(x_train, y_train)
        y_predict = lm.predict(x_test) 
        train_score = lm.score(x_train, y_train)
        test_score = lm.score(x_test, y_test)
        if train_score >= 0.70 and test_score >= 0.75:
            print("variable:", var, "outcome:", outcome)
            print("train score:", train_score, "test score:", test_score)
            
            plt.scatter(y_test, y_predict, alpha = 0.5)
            plt.title("Actual " + str(outcome) + " vs Predicted " + str(outcome) + " Single Feature: " + str(var))
            plt.xlabel(outcome)
            plt.ylabel("Predicted " + str(outcome))
            plt.show()
            plt.clf()

## BreakPointsOpportunities is the best way to predict wins
        

####################################
## Two Feature Linear Regressions ##
####################################

x = tennis[["BreakPointsOpportunities", "Aces"]]
y = tennis[["Wins"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2)

lm.fit(x_train, y_train)
y_predict = lm.predict(x_test)

plt.scatter(y_test, y_predict, alpha = 0.5)
plt.title("Wins vs Predicted Wins: 2 Features BreakPointsOpportunities & Aces")
plt.xlabel("Actual Wins")
plt.ylabel("Predicted Wins")
plt.show()
plt.clf()


x = tennis[["BreakPointsFaced", "ServiceGamesPlayed"]]
y = tennis[["Losses"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2)

lm.fit(x_train, y_train)
y_predict = lm.predict(x_test)

plt.scatter(y_test, y_predict, alpha = 0.5)
plt.title("Losses vs Predicted Losses: 2 Features BreakPointFaced & ServiceGamesPlayed")
plt.xlabel("Actual Losses")
plt.ylabel("Predicted Losses")
plt.show()
plt.clf()


x = tennis[["DoubleFaults", "ReturnGamesPlayed"]]
y = tennis[["Winnings"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2)

lm.fit(x_train, y_train)
y_predict = lm.predict(x_test)

plt.scatter(y_test, y_predict, alpha = 0.5)
plt.title("Winnings vs Predicted Winnings: 2 Features DoubleFaults, ReturnedGamesPlayed")
plt.xlabel("Actual Winnings")
plt.ylabel("Predicted Winnings")
plt.show()
plt.clf()


x = tennis[["ServiceGamesPlayed", "ReturnGamesPlayed"]]
y = tennis[["Wins"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2)

lm.fit(x_train, y_train)
y_predict = lm.predict(x_test)

plt.scatter(y_test, y_predict, alpha = 0.5)
plt.title("Wins vs Predicted Wins: 2 Feature ServiceGamesPlayed & ReturnGamesPlayed")
plt.xlabel("Actual Wins")
plt.ylabel("Predicted Wins")
plt.show()
plt.clf()


#########################################
## Multiple Feature Linear Regressions ##
#########################################

x = tennis[["FirstServe", "FirstServePointsWon", "FirstServeReturnPointsWon", "SecondServePointsWon", "SecondServeReturnPointsWon", "Aces", "BreakPointsConverted", "BreakPointsFaced", "BreakPointsOpportunities", "BreakPointsSaved", "DoubleFaults", "ReturnGamesPlayed", "ReturnGamesWon", "ReturnPointsWon", "ServiceGamesPlayed", "ServiceGamesWon", "TotalPointsWon"]]
y = tennis[["Winnings"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2)

lm.fit(x_train, y_train)
y_predict = lm.predict(x_test)

plt.scatter(y_test, y_predict, alpha = 0.5)
plt.title("Winnings vs Predicted Winnings")
plt.show()
plt.clf()


#############################################################
## Multiple Feature Linear Regression for Winnings by Year ##
#############################################################

print(tennis.Year.unique())

## Data by Year
tennis09 = tennis[tennis["Year"] == 2009]
tennis10 = tennis[tennis["Year"] == 2010]
tennis11 = tennis[tennis["Year"] == 2011]
tennis12 = tennis[tennis["Year"] == 2012]
tennis13 = tennis[tennis["Year"] == 2013]
tennis14 = tennis[tennis["Year"] == 2014]
tennis15 = tennis[tennis["Year"] == 2015]
tennis16 = tennis[tennis["Year"] == 2016]
tennis17 = tennis[tennis["Year"] == 2017]

x = tennis09[["DoubleFaults", "BreakPointsOpportunities", "ServiceGamesPlayed"]]
y = tennis09[["Winnings"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2)

print("x train and test shape:", x_train.shape, x_test.shape)
print("y train and test shape:", y_train.shape, y_test.shape)

lm.fit(x_train, y_train)
y_predict = lm.predict(x_test)

plt.scatter(y_test, y_predict, alpha = 0.5)
plt.title("2009: Winnings vs Predicted Winnings: 3 Features")
plt.xlabel("Actual Winnings")
plt.ylabel("Predicted Winnings")
plt.show()
plt.clf()


x = tennis12[["DoubleFaults", "BreakPointsOpportunities", "ServiceGamesPlayed"]]
y = tennis12[["Winnings"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2)

print("x train and test shape:", x_train.shape, x_test.shape)
print("y train and test shape:", y_train.shape, y_test.shape)

lm.fit(x_train, y_train)
y_predict = lm.predict(x_test)

plt.scatter(y_test, y_predict, alpha = 0.5)
plt.title("2012: Winnings vs Predicted Winnings: 3 Features")
plt.xlabel("Actual Winnings")
plt.ylabel("Predicted Winnings")
plt.show()
plt.clf()


x = tennis15[["DoubleFaults", "BreakPointsOpportunities", "ServiceGamesPlayed", "Aces"]]
y = tennis15[["Winnings"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2)

print("x train and test shape:", x_train.shape, x_test.shape)
print("y train and test shape:", y_train.shape, y_test.shape)

lm.fit(x_train, y_train)
y_predict = lm.predict(x_test)

plt.scatter(y_test, y_predict, alpha = 0.5)
plt.title("2015: Winnings vs Predicted Winnings: 4 Features")
plt.xlabel("Actual Winnings")
plt.ylabel("Predicted Winnings")
plt.show()
plt.clf()


x = tennis17[["DoubleFaults", "BreakPointsOpportunities", "ServiceGamesPlayed", "Aces", "ReturnGamesPlayed", "BreakPointsFaced"]]
y = tennis17[["Winnings"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2)

print("x train and test shape:", x_train.shape, x_test.shape)
print("y train and test shape:", y_train.shape, y_test.shape)

lm.fit(x_train, y_train)
y_predict = lm.predict(x_test)

plt.scatter(y_test, y_predict, alpha = 0.5)
plt.title("2017: Winnings vs Predicted Winnings: 6 Features")
plt.xlabel("Actual Winnings")
plt.ylabel("Predicted Winnings")
plt.show()
plt.clf()