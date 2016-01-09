import argparse
import requests
import json
import sqlite3 as sql
import re
import datetime as dt
import pandas as pd
from progress.bar import Bar

api_key = open('api_key.txt', 'r').read().strip()

url = 'http://api.sportsdatabase.com/nba/query.json?sdql='

def season_team_game_stats(select, season=2014):
    """
    Passes points parameters select to SDQL query in a given season.
    Documentation on SDQL queries available here: http://www.sdql.com/
    Outputs to data/NBAseasonYYYY.json
    """

    today = dt.date.today().strftime('%Y%m%d')

    param_string = ','.join(select)

    param_string = param_string.replace(',','%2C')
    param_string = param_string.replace(':','%3A')
    param_string = param_string.replace('=','%3D')

    call = url + param_string + '%40season%3D{0}'.format(season) + \
           '&date<={}'.format(today) + '&output=json&api_key='\
           + api_key #administrative stuff

    r = requests.get(call)

    with open('data/NBAseason{0}.json'.format(season),'w') as f:
        f.write(r.content[14:-3].replace('\'', '"'))

def data_from_json(season=2014):
    """
    Reads jsons from data/NBAseasonYYYY.json
    into a pandas DataFrame
    """

    with open('data/NBAseason{0}.json'.format(season)) as f:
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
        out['t.total_rebounds'] = out[[
            't.offensive_rebounds','t.defensive_rebounds'
        ]].sum(axis=1)

    out.set_index(['season','team','date'],inplace = True)
    out.dropna(thresh=10)
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='grabs relevant data for specified seasons'
    )
    parser.add_argument(
        '--seasons', '-s', nargs='*', type=int, default=2015,
        help='seasons to grab data for'
    )
    args = parser.parse_args()
    seasons = sorted(args.seasons)


    sel = ['date','team','o:team','game_number','t:points','to:points', #indexing and general
           't:minutes','t:three pointers made','t:assists','t:field goals made',
           't:turnovers','t:field goals attempted','t:free throws attempted','t:free throws made',
           't:defensive rebounds','t:offensive rebounds','t:steals','t:blocks','t:fouls',#PER volume stats
           'to:defensive rebounds','to:offensive rebounds','to:free throws attempted','to:turnovers', #PER pace stats
           't:three pointers attempted','t:LSP','to:field goals attempted','to:field goals made',
           'to:three pointers made', 'to:three pointers attempted',#for clustering
           't:ats margin','t:site']#gambling stats

    print 'Fetching game data...'
    bar = Bar(width=40)
    for season in bar.iter(seasons):
        season_team_game_stats(sel, season)

    data = pd.concat(
        [data_from_json(season) for season in seasons]
    )
    # assumes 0 minute games were a normal length and just errors in
    # data entry
    data['t.minutes'].replace(0,240,inplace = True)

    data.to_pickle('data/team_data_{0}_{1}.pkl'.format(
        seasons[0], seasons[-1]
    ))
    data.to_csv('data/team_data_{0}_{1}.csv'.format(
        seasons[0], seasons[-1]
    ))
    with sql.connect('data/nba_data.db') as nba_data:
        cur = nba_data.cursor()
        existence_check = [name[0] for name in cur.execute(
            'SELECT name FROM sqlite_master WHERE type="table";'
        ).fetchall()]
        if 'game_data' in existence_check:
            temp = pd.read_sql(
                'SELECT * FROM game_data;', nba_data,
                index_col=['season', 'team', 'date']
            )
            to_concat = data.loc[~data.index.isin(temp.index)]
            pd.concat([temp,to_concat]).to_sql(
                'game_data', nba_data, if_exists='replace'
            )
        else:
            data.to_sql('game_data', nba_data)
