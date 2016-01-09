import json
import re
import sqlite3 as sql
import time
import datetime as dt
import requests
import pandas as pd

api_key = open('api_key.txt', 'r').read().strip()

url = 'http://api.sportsdatabase.com/nba/query.json?sdql='

def _get_season():
    tmp = dt.date.today()
    return tmp.year if tmp.month >= 7 else tmp.year - 1

def upcoming_games():
    """
    Reads gains in the next 8 days from sportsdatabase and dumps into
    data/upcoming_games.json
    """

    today = dt.date.today()
    future = today + dt.timedelta(days=8)
    today = today.strftime('%Y%m%d')
    future = future.strftime('%Y%m%d')
    param_string = 'team,o:team,date,site'

    param_string = param_string.replace(',','%2C')
    param_string = param_string.replace(':','%3A')
    param_string = param_string.replace('=','%3D')

    call = (
        url + param_string +
        '%40date<%3D{FUTURE}+and+date>%3D{TODAY}'
        '&output=json&api_key=' + api_key #administrative stuff
    ).format(FUTURE=future, TODAY=today)

    r = requests.get(call)

    with open('data/upcoming_games.json', 'w') as f:
        f.write(r.content[14:-3].replace('\'', '"'))


def data_from_json(season):
    """
    Reads jsons from data/upcoming_games.json
    into a pandas DataFrame
    """

    with open('data/upcoming_games.json') as f:
        j = json.loads(f.read())
        out = pd.DataFrame(columns = j['headers'])
        for i, c in enumerate(out.columns):
            out[c] = j['groups'][0]['columns'][i]
        out['season'] = season
        out.rename(columns={
            c:re.sub(' ', '_', c) for c in out.columns
        }, inplace=True)
        out.rename(columns={
            c:re.sub('\:', '.', c) for c in out.columns
        }, inplace=True) #removing : for patsy
        out.drop([i for i in out.index if out.loc[i,'site'] != 'home'],
                 inplace=True)
    out['date'] = out['date'].apply(
        lambda x: pd.to_datetime(str(x), format='%Y%m%d')
    )

    return out.set_index(['season','team','date']).drop('site', axis=1)

def main():
    upcoming_games()
    return data_from_json(_get_season())

if __name__ == '__main__':
    main().to_csv('data/upcoming_games.csv')
