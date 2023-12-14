
# League of Legends Win Chance Prediction

### ML model to predict the outcome of a League of Legends match based on champion selection

## Introduction

League of Legends, often abbreviated as LoL, is a popular online
multiplayer video game. It\'s a competitive 5 versus 5 team-based game
in which players control unique champions with special abilities and
work together to defeat the opposing team. The main objective is to
destroy the enemy team\'s Nexus, a structure in their base, while
defending your own. It combines elements of strategy, teamwork, and
individual skill and is known for its strategic depth and fast-paced
action. League of Legends is played by millions of players worldwide and
has a thriving esports scene with professional leagues and tournaments.

In the competitive environment of League of Legends, players are always
looking for ways to improve their chances of winning. Since it\'s a
strategy game, one key element affecting a team\'s success is the mix of
champions they pick. Our aim is to create a model that helps players
make better decisions about champion selection and team composition by
predicting the likelihood of each team winning based on their chosen
champions. This also enables the most dedicated players to dodge an
unfavorable matchup before the game begins in such a case where the
prediction of their chances of winning are looking less than good.

The information about the match is limited to just the champions picked
before the game actually begins, so we are going to be using only this
information for training our model.

## Project
For a detailed overview, please look at our [notebook file](/project.ipynb).

## Goals
- train neural network to predict outcome of League of Legends games based only on champions picked
- use trained network to estimate win chance during champion select
- use trained network to calculate best pick for each stage of champion select

## Requirements
The requirements can be found in env.yml.

## Example
![champ select](data/notebook/img/champ_sel1.png)

### Win chance
```python
from lol_prediction import LoLPredictor

predictor = LoLPredictor(lol_transformer_8_8)

blue = ["VelKoz", "Jhin", "Taliyah"]
red = ["Xayah", "Graves", "Malzahar", "LeBlanc"]

# predict blue side win chance
predictor.win_chance(blue, red)
```
Output:
```
1/1 [==============================] - 0s 32ms/step
0.5423562526702881
```

### Best Picks
```python
# predict next best champion (for blue side)
_=predictor.best_pick(blue, red)
```
Output:
```
5/5 [==============================] - 0s 13ms/step
5 best picks for
['Velkoz', 'Jhin', 'Taliyah', 'null', 'null']
vs
['Xayah', 'Graves', 'Malzahar', 'Leblanc', 'null']
Shyvana: 57.66%
Warwick: 57.53%
Renata: 57.5%
Sejuani: 57.24%
Amumu: 57.19%
```

```python
# best picks from list
my_champs = ["KhaZix", "LeeSin", "Gragas", "Ivern"]
_=predictor.best_pick(blue, red, my_champs)
```
Output:
```
1/1 [==============================] - 0s 29ms/step
5 best picks for
['Velkoz', 'Jhin', 'Taliyah', 'null', 'null']
['Xayah', 'Graves', 'Malzahar', 'Leblanc', 'null']
Gragas: 54.78%
Khazix: 54.73%
Ivern: 53.37%
LeeSin: 50.88%
```

## Getting started
### Pretrained

### Training
