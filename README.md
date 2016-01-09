NBA Spread Predictor
====================

This repo contains code for predicting whether the home team covers the point
spread based on game data from 2002 to present. There are four important
scripts:

- `game_data.py`: contains functions for obtaining historical data from [The
    Sports Database](http://sportsdatabase.com/)
- `upcoming_games.py`: contains functions for obtaining matchups for the
    upcoming week, also from The Sports Database
- `pred_NBA.py`: contains two classes, `FeatureMaker` and `SpreadPredictor`,
    that are used to predict whether the home team covers
- `runner.py`: runs the whole show.

## Running

You'll need three directories to run without errors:

- `prediction_info`: for storing some metadata about predictions on the true
    out-of-sample dataset
- `output`: for storing predictions from the upcoming week
- `data`: because `game_data.py` stores a bunch of jsons/a sql db here when you run it

Then create a virtualenv, install dependencies, and run, e.g., `python runner.py
rforest`. Note: this takes **a really long time**. The first time, you'll
probably want to find the part of `pred_NBA.py` under `RUNTIME REDUCER` and
uncomment out the lines that make the `GridSearch` stop sooner, to see how
things look.

## Results

I've found that on average the random forest classifier scores about 60%
accurate on the true OOS set and I've been around that in real life with it as
well.

## Disclaimer, because I don't know if I need one

I am not a Registered Investment Advisor, Broker/Dealer, Financial
Analyst, Financial Bank, Securities Broker or Financial Planner. The Information
in this repository is provided for information purposes only. The Information is not
intended to be and does not constitute financial advice or any other advice, is
general in nature and not specific to you. Before using my model
to make a betting decision, you should undertake your own due
diligence. I am not responsible for any investment decision made by you.
You are responsible for your own investment research and investment decisions.
