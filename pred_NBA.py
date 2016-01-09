import time
import sys
from itertools import product
import datetime as dt
import sqlite3 as sql
from progress.bar import Bar
import pandas as pd
import numpy as np
from patsy import dmatrix, dmatrices
from sklearn.cross_validation import KFold, train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

import upcoming_games as ug

def run(season, method):
    """
    Runs a SpreadPredictor with variation in pen_params.
    """

    sp = SpreadPredictor(season=season, method=method)
    sp.tune_model()
    sp.raw_oos_pred().to_csv(
        'prediction_info/preds_{}.csv'.format(
            dt.datetime.today().strftime('%Y%m%d')
        )
    )
    return sp.new_predictions()
    # sp.plot_test_error(pen_params=params[method])


class FeatureMaker(object):
    """
    Class for classifying team based on offensive and defensive
    rate statistics.
    """

    def __init__(self, season, date=dt.date.today()):
        self.season = season
        self.as_of = date
        self.db = sql.connect('data/nba_data.db')
        self.cur = self.db.cursor()
        self.cur.execute(
            'SELECT DISTINCT season FROM game_data;'
        )
        self.seasons = [t[0] for t in self.cur.fetchall()]
        assert self.season in self.seasons

        # get all games that have been played
        self.data = pd.read_sql(
            """
            SELECT * FROM game_data;
            """,
            self.db
        ).sort().dropna(how='any')
        self.data = self.data[self.data['season'] <= self.season]

        self.data['date'] = [
            self.make_date(d) for d in self.data['date']
            ]
        self.first_day = self.data[
            self.data['season'] == self.season
        ]['date'].min()
        self.data.set_index(['season', 'team', 'date'], inplace=True)
        self.data = self.data.groupby(
            level=self.data.index.names
        ).first()
        self.data.sort(inplace=True)

        self.poss_table = self.calc_possessions().groupby(
            level=['season', 'team', 'date']
        ).first()
        self.o_rate_stats, self.d_rate_stats = self.rate_stats()
        self.o_rate_stats.sort(inplace=True)
        self.d_rate_stats.sort(inplace=True)

        self.rolling_o = self.o_rate_stats.groupby(
            level=['season', 'team']
        ).apply(pd.rolling_mean, 15, 1)
        self.rolling_d = self.d_rate_stats.groupby(
            level=['season', 'team']
        ).apply(pd.rolling_mean, 15, 1)
        self.o_types = pd.DataFrame(index=self.o_rate_stats.index)
        self.d_types = pd.DataFrame(index=self.d_rate_stats.index)

        self.o_fitted = None
        self.d_fitted = None
        self.o_cluster_centers = None
        self.d_cluster_centers = None

        self.pers = self.all_pers()

    def get_max_date(self, s, table):
        """
        Returns latest date with a value from table in self.db
        """

        self.cur.execute(
            """
            SELECT MAX(date) FROM {} WHERE season=?;
            """.format(table),
            (s,)
        )

        returned = self.cur.fetchall()
        if len(returned) == 0:
            return False

        dmax = time.strptime(
            returned[0][0],
            '%Y-%m-%d %H:%M:%S'
        )

        return dt.datetime(*dmax[:3])

    def make_date(self, date_int):
        """
        Returns a datetime object from yyyymmdd formatted date string
        or date int.
        """

        return dt.datetime(
            int(str(date_int)[:4]),
            int(str(date_int)[4:6]),
            int(str(date_int).strip('.0')[6:])
        )

    def calc_possessions(self):
        """
        Calculates number of opponent possessions in each game
        through self.as_of.
        """

        working = pd.read_sql(
            """
            SELECT season, team, game_number, date, "o.team",
            "t.field_goals_attempted", "t.offensive_rebounds",
            "t.turnovers", "t.free_throws_attempted"
            FROM game_data;
            """,
            self.db
        )

        working['date'] = working['date'].apply(self.make_date)
        working.set_index(['season', 'team', 'date'], inplace=True)
        working.sort(inplace=True)
        working['poss'] = working['t.field_goals_attempted'] - \
                          working['t.offensive_rebounds'] + \
                          working['t.turnovers'] + \
                          working['t.free_throws_attempted'] * 0.44

        return working[['poss', 'o.team']]

    def rate_stats(self):
        """
        Calculates rate stats for each team through self.as_of.

        Offensive rate stats are all per 100 possessions.
        Defensive rate stats are a mix of per 100 possessions and
        of success rates.

        OFFENSE
        - assist rate
        - three point attempt rate
        - free throw attempt rate
        - field goal attempt rate

        DEFENSE
        - steals + blocks
        - three point attempts allowed
        - fouls
        - field goal percentage allowed
        """

        idx = pd.IndexSlice
        working = pd.read_sql(
            """
            SELECT season, team, date, "o.team",
            "t.assists", "t.three_pointers_attempted",
            "t.free_throws_attempted", "t.field_goals_attempted",
            "t.steals", "t.blocks", "t.fouls", "to.three_pointers_attempted",
            "to.field_goals_made", "to.field_goals_attempted" FROM game_data;
            """,
            self.db
        )

        working['date'] = working['date'].apply(self.make_date)
        working.set_index(['season', 'team', 'date'], inplace=True)
        working.sort(inplace=True)

        print 'Calculating rate stats...'

        bar = Bar(
            width=40,
            suffix='%(percent)d%%'
        )

        for i in bar.iter(working.index):
            working.loc[i, 'o.poss'] = self.poss_table.loc[
                idx[i[0], self.poss_table.loc[i, 'o.team'], i[2]],
                'poss'
            ]

        o_stats = pd.DataFrame(
            {
                'assist_rate': working['t.assists'] /
                               self.poss_table['poss'] * 100,
                '3p_attempt_rate': working['t.three_pointers_attempted'] /
                                   self.poss_table['poss'] * 100,
                'FT_attempt_rate': working['t.free_throws_attempted'] /
                                   self.poss_table['poss'] * 100,
                'FG_attempt_rate': working['t.field_goals_attempted'] /
                                   self.poss_table['poss'] * 100
            }
        ).sort()

        d_stats = pd.DataFrame(
            {
                'st+bl_rate': (working['t.steals'] + working['t.blocks']) /
                              working['o.poss'] * 100,
                '3PA_allowed': working['to.three_pointers_attempted'] /
                               working['o.poss'] * 100,
                'foul_rate': working['t.fouls'] / working['o.poss'] * 100,
                'FG%_allowed': working['to.field_goals_made'] /
                               working['to.field_goals_attempted'] * 100
            }
        ).sort()

        return [o_stats, d_stats]

    def o_centers(self, n=4, centers=None):
        """
        Returns (season, team, date)-indexed offensive types
        for this season (or season represented by data) by self.as_of
        """

        idx = pd.IndexSlice
        model = KMeans(n_clusters=n,
                       init=centers if centers else 'k-means++'
                       )

        data = self.o_rate_stats.loc[
            idx[self.season, :, self.first_day:self.as_of]
        ].dropna().groupby(level='team').mean()

        fitted = model.fit(data)

        self.o_cluster_centers = fitted.cluster_centers_
        self.o_fitted = fitted

    def d_centers(self, n=4, centers=False):
        """
        Returns (season, team, date)-indexed defensive types
        for  this season (or season represented by data) by self.as_of
        """

        idx = pd.IndexSlice

        model = KMeans(n_clusters=n,
                       init=centers if centers else 'k-means++'
                       )

        data = self.d_rate_stats.loc[
            idx[self.season, :, self.first_day:self.as_of]
        ].dropna().groupby(level='team').mean()

        fitted = model.fit(data)

        self.d_cluster_centers = fitted.cluster_centers_
        self.d_fitted = fitted

    def _classify(self, fitted, data, name):
        """
        Generic function for classififying data based on given
        cluster centers.
        """

        out = pd.Series(
            fitted.predict(data), index=data.index,
            name=name,
            dtype='category'
        )
        return out.groupby(level=out.index.names).first()

    def o_classify(self):
        """
        Classifies teams based on offensive data.
        """

        print 'Classifying teams into offensive buckets...'

        if not self.o_fitted:
            self.o_centers()

        t_types = self._classify(
            fitted=self.o_fitted,
            data=self.rolling_o.dropna(),
            name='otype'
        )

        vals = product(
            t_types.index.get_level_values('season').unique(),
            t_types.index.get_level_values('team').unique(),
            t_types.index.get_level_values('date').unique()
        )
        new_ind = pd.MultiIndex.from_tuples(
            list(vals),
            names=t_types.index.names,
        )
        self.o_types = t_types.reindex(new_ind)

    def d_classify(self):
        """
        Classifies teams based on defensive data.
        """

        print 'Classifying teams into defensive buckets...'

        if not self.d_fitted:
            self.d_centers()

        t_types = self._classify(
            fitted=self.d_fitted,
            data=self.rolling_d.dropna(),
            name='dtype'
        )

        vals = product(
            t_types.index.get_level_values('season').unique(),
            t_types.index.get_level_values('team').unique(),
            t_types.index.get_level_values('date').unique()
        )
        new_ind = pd.MultiIndex.from_tuples(
            list(vals),
            names=['season', 'team', 'date']
        )
        self.d_types = t_types.reindex(new_ind)

    def season_per_adjusters(self, s):
        """
        Returns factor, VOP, and DRB%, as described in layer efficiency
        rating calculation taken from:
        http://www.basketball-reference.com/about/per.html
        """

        idx = pd.IndexSlice
        working = self.data.xs(s, level='season')

        dates = working.index.get_level_values('date').unique()
        date_ind = pd.MultiIndex.from_product(
            [
                [s],
                dates
            ],
            names=['season', 'date']
        )

        factors = pd.Series(
            [(2 / 3.) - (
                0.5 * working.loc[idx[:, :d], 't.assists'].sum() /
                working.loc[idx[:, :d], 't.field_goals_made'].sum()
            ) / (
                2 * working.loc[idx[:, :d], 't.field_goals_made'].sum() /
                working.loc[idx[:, :d], 't.free_throws_made'].sum()
            ) for d in dates
             ], index=date_ind)

        vops = pd.Series([
            working.loc[idx[:, :d], 't.points'].sum() /
            (
                working.loc[idx[:, :d], 't.field_goals_attempted'].sum() -
                working.loc[idx[:, :d], 't.offensive_rebounds'].sum() +
                working.loc[idx[:, :d], 't.turnovers'].sum() +
                0.44 * working.loc[idx[:, :d], 't.free_throws_attempted'].sum()
            ) for d in dates
        ], index=date_ind)

        drb_percs = pd.Series([
            (
                working.loc[idx[:, :d], 't.total_rebounds'].sum() -
                working.loc[idx[:, :d], 't.offensive_rebounds'].sum()
            ) / working.loc[idx[:, :d], 't.total_rebounds'].sum()
            for d in dates
        ], index=date_ind)

        poss_working = self.poss_table.xs(s, level='season')

        # based on layout of data, did not separately add up
        # team and opponent possessions, as is done on BR, Pace:
        # http://www.basketball-reference.com/about/glossary.html#pace
        lg_paces = pd.Series([
            (
                poss_working.loc[idx[:, :d], 'poss'].sum() /
                (2 * working.loc[idx[:, :d], 't.minutes'].sum() / 5)
            ) for d in dates
        ], index=date_ind)

        return factors, vops, drb_percs, lg_paces

    def per(self, s, team, date,
            factors=None,
            vops=None,
            drb_percs=None,
            lg_paces=None):
        """
        Calculates team efficiency rating for a team in a season as of
        a certain date.

        Uses calculation from basketball-reference:
        http://www.basketball-reference.com/about/per.html

        Note: if any of factors, vops, drb_percs, or lg_pace is not
        provided, the function will calculate all four. Probably if you're
        doing this elsewhere you want that not to happen for efficiency.
        """

        idx = pd.IndexSlice
        out_ind = pd.MultiIndex.from_tuples(
            [(s, team, date)],
            names=['season', 'team', 'date']
        )

        # handles date before first game of season
        if date < self.data.loc[idx[s, team]].index[0]:
            return pd.DataFrame(
                index=out_ind,
                columns=['aPER']
            )

        for par in [factors, vops, drb_percs, lg_paces]:
            if par is None:
                factors, vops, drb_percs, lg_pace = \
                    self.season_per_adjusters(s)
                break

        working_team = self.data.loc[idx[s, team, :date]]
        try:
            assert len(working_team) > 0
        except AssertionError:
            print s, team, date
            raise
        assert len(self.poss_table.loc[idx[s, team, :date], 'o.team']) > 0
        assert len(working_team.index.get_level_values('date').tolist()) > 0
        opp_ind = pd.MultiIndex.from_tuples(
            zip(
                [s] * len(working_team),
                self.poss_table.loc[idx[s, team, :date], 'o.team'],
                working_team.index.get_level_values('date').tolist()
            )
        )
        working_opp = self.poss_table.loc[opp_ind]

        factor = factors[idx[s, date]]
        vop = vops[idx[s, date]]
        drb_perc = drb_percs[idx[s, date]]
        lg_pace = lg_paces[idx[s, date]]
        team_poss = self.poss_table.loc[
            idx[s, team, :date], 'poss'
        ].sum()
        opp_poss = working_opp['poss'].sum()
        team_pace = (team_poss + opp_poss) / \
                    (2 * self.data.loc[
                        idx[s, team, :date], 't.minutes'
                    ].sum() / 5)

        lg_ft = float(self.data.loc[idx[s, :, :date],
                                    't.free_throws_made'].sum())
        lg_fta = float(self.data.loc[idx[s, :, :date],
                                     't.free_throws_attempted'].sum())
        lg_pf = float(self.data.loc[idx[s, :, :date],
                                    't.fouls'].sum())

        uPER = (1 / float(working_team['t.minutes'].sum())) * \
               (
                   working_team['t.three_pointers_made'].sum() +
                   (2 / 3.) * working_team['t.assists'].sum() +
                   (
                       (
                           2 - factor * working_team['t.assists'].sum() /
                           working_team['t.field_goals_made'].sum()
                       ) *
                       working_team['t.field_goals_made'].sum()
                   ) +
                   (
                       working_team['t.free_throws_made'].sum() * 0.5 *
                       (
                           1 + (
                               1 - working_team['t.assists'].sum() /
                               float(
                                   working_team['t.field_goals_made'].sum()
                               )
                           ) +
                           (2 / 3.) * working_team['t.assists'].sum() /
                           float(working_team['t.field_goals_made'].sum())
                       )
                   ) -
                   vop * working_team['t.turnovers'].sum() -
                   vop * drb_perc * (
                       working_team['t.field_goals_attempted'].sum() -
                       working_team['t.field_goals_made'].sum()
                   ) -
                   vop * 0.44 * (
                       0.44 + (0.56 * drb_perc) *
                       working_team['t.free_throws_attempted'].sum() -
                       working_team['t.free_throws_made'].sum()
                   ) +
                   vop * (1 - drb_perc) * (
                       working_team['t.total_rebounds'].sum() -
                       working_team['t.offensive_rebounds'].sum()
                   ) +
                   vop * drb_perc * working_team[
                       't.offensive_rebounds'
                   ].sum() +
                   vop * working_team['t.steals'].sum() +
                   vop * drb_perc * working_team['t.blocks'].sum() -
                   working_team['t.fouls'].sum() * (
                       (lg_ft / float(lg_pf)) - 0.44 *
                       (lg_fta / float(lg_pf)) * vop
                   )
               )
        # because this just calculates for one team, we don't yet have
        # enough to calculate the normalized-to-15 PER that Hollinger/BR
        # use. That can be done separately and more efficiently once
        # everything is done. See method normed_PERs

        aPER = (lg_pace / team_pace) * uPER
        assert aPER > 0
        return pd.DataFrame(
            {'aPER': [aPER]},
            index=out_ind
        )

    def team_pers(self, s):
        """
        Returns (season, team, date)-indexed team efficiency ratings
        for dataset through self.as_of for season.

        Team efficiency rating calculated as player efficiency rating,
        but for whole teams. Player efficiency rating calculation taken
        from:
        http://www.basketball-reference.com/about/per.html
        """

        idx = pd.IndexSlice
        try:
            working = self.data.xs(s, level='season')
        except KeyError:
            return pd.DataFrame()
        if len(working.dropna(how='any')) == 0:
            return pd.DataFrame()
        dates = working.index.get_level_values('date').unique()
        teams = working.index.get_level_values('team').unique()
        factors, vops, drb_percs, lg_pace = self.season_per_adjusters(
            s
        )

        # limits dates to those not already present in database
        # if latest handles case where table exists but season has not
        # yet been processed
        try:
            yearscheck = pd.read_sql(
                'SELECT * FROM aPERs WHERE season={};'.format(s),
                self.db
            )
            if s in yearscheck['season']:
                latest = self.get_max_date(s, 'aPERs')
                if latest:
                    dates = dates[dates > latest]
        except sql.OperationalError:
            pass

        if len(dates) == 0:
            out = pd.read_sql(
                """
                SELECT * FROM aPERs WHERE season={};
                """.format(s),
                self.db
            )

            out['date'] = out['date'].apply(
                lambda x: dt.datetime(
                    *time.strptime(x, '%Y-%m-%d %H:%M:%S')[:6]
                )
            )

            return out.set_index(['season', 'team', 'date'])

        bar = Bar(
            width=40,
            suffix='%(percent)d%%'
        )

        print 'Calculating PERs for season {}...'.format(s)

        out = pd.concat(
            [self.per(s, t, d, factors, vops, drb_percs, lg_pace)
             for t, d in bar.iter(
                [(te, da) for te in teams for da in dates]
            )]
        ).sort()

        out.to_sql('aPERs', self.db, if_exists='append')

        return out

    def normed_pers(self, s):
        """
        Normalizes PER calculations within a season to 15.
        See team_pers, pers for method of calculating PER.
        """

        pass

    def all_pers(self):
        """
        Calculates PER for the whole dataset. See per, team_pers for
        method on calculating PER.
        """


        bar = Bar(width=40)
        print 'Calculating PERs...'
        return pd.concat(
            [self.team_pers(s) for s in bar.iter(self.seasons)]
        ).groupby(level=self.data.index.names).first()


class SpreadPredictor(object):
    """
    Predicts spread based on bootstrapped historical spreads given
    some input data.
    """

    def __init__(self, as_of=dt.date.today(), season=2014, method='svr'):
        self.as_of = as_of
        self.fm = FeatureMaker(season, date=as_of)
        self.method = method
        self.models = {
            'svr': SVR,
            'svc': SVC,
            'rforest': RandomForestClassifier,
            'ridge': Ridge
        }
        self.model = self.models[method]

        # generate all of the data
        self.fm.o_classify()
        self.fm.d_classify()

        self.data = pd.concat([
            self.fm.o_types,
            self.fm.d_types,
            self.fm.pers,
            self.fm.data[
                ['t.ats_margin', 't.points', 'to.points',
                 't.site', 'o.team']
            ]
        ], axis=1)
        self.data['real_spread'] = self.data[
                                       'to.points'
                                   ] - self.data['t.points']
        self.data['cover'] = (
            self.data['real_spread'] < self.data['t.ats_margin']
        ).astype(int)
        self.data = self.data[self.data['t.site'] == 'home']
        print self.data.head()

        # read upcoming games to data
        self.to_predict = ug.main().sort().reindex(
            columns=['o.team', 'aPER', 'otype', 'dtype']
        ).reset_index(level='date')
        self.to_predict.index = self.to_predict.index.droplevel(
            ['season']
        )
        self.to_predict.update(
            self.data.groupby(level='team')[
                ['aPER', 'otype', 'dtype']
            ].last()
        )
        self.to_predict.set_index('date', inplace=True, append=True)
        self.outcome = 'real_spread' if method in ['svr', 'ridge'] else \
            'cover'
        self.predictors = [
            'aPER',
            'C(otype)',
            'C(dtype)'
        ]
        self.bestmod = None

        self.dirty_y, self.clean_y, self.dirty_X, self.clean_X = \
            self.big_split()

    def big_split(self):
        """
        Returns training data (big chunk) and test data that will not
        be used for any of the training. Note that this train/test split
        is not what's used for model selection,
        """

        add_on = ' + C(otype):C(dtype)' if self.method != 'rforest' else \
            ''

        y, X = dmatrices(
            self.outcome + ' ~ ' + ' + '.join(
                self.predictors
            ) + add_on,
            data=self.data,
            return_type='dataframe'
        )

        return train_test_split(y, X, test_size=0.2)

    def k_fold_results(self, **kwargs):
        """
        Takes a dict of keyword arguments and trains on k-folded
        dirty_y and dirty_X from above. **kwargs passed to initialization
        of model being trained/tested

        GOTCHA: Does no name checking to make sure **kwargs keys
        actually appear in parameters of target model. User beware until
        I someday figure out how to fix that.
        """

        kf = KFold(
            n=len(self.dirty_X),
            n_folds=8,
            shuffle=True
        )
        model = self.model(**kwargs)
        X = self.dirty_X
        y = np.squeeze(self.dirty_y)

        return cross_val_score(model, X, y, cv=kf, n_jobs=-1)

    def tune_model(self):
        """
        Tunes self.model using a GridSearch.
        """

        grid = {
            'ridge': ParameterGrid({
                'alpha': [i / 2. for i in range(1, 21)],
            }),
            'svc': ParameterGrid({
                'C': [i / 2. for i in range(1, 21)],
                'gamma': [i / 10. for i in range(1, 10)]
            }),
            'svr': ParameterGrid({
                'C': [i / 2. for i in range(1, 21)],
                'epsilon': [i / 10. for i in range(11)]
            }),
            'rforest': ParameterGrid({
                'n_estimators': range(25, 501, 25),
                'max_depth': range(2, 10),
                'min_samples_split': range(5, 101, 5)
            })
        }[self.method]

        best = {'params': None, 'score': 0}

        bar = Bar(message='Searching...',
                  width=40,
                  suffix='%(percent)d%%')
        for params in bar.iter(grid):
            # RUNTIME REDUCER
#            if np.random.uniform() > 0.975:
#                break
            score = cross_val_score(
                self.model(**params),
                self.dirty_X,
                np.squeeze(self.dirty_y),
                cv=8,
                n_jobs=4
            ).mean()
            if score > best['score']:
                best.update({
                    'params': params,
                    'score': score
                })

        bestmod = self.model(
            n_jobs=-1,
            **best['params']
        )
        bestmod.fit(
            self.dirty_X,
            np.squeeze(self.dirty_y)
        )
        self.bestmod = bestmod

    def plot_test_error(self, pen_params):
        """
        Generates test error plot over varying hyperparameters.
        **params is passed to cross_val_score as **kwargs. Each key-value
        pair in **params should be

        param: np array of values
        """

        def _non_forest_plot(self, pen_params):
            bar = Bar(
                width=40,
                suffix='%(percent)d%%'
            )
            values = pen_params.values()[0]
            print 'Getting errors for {}...'.format(self.method)
            errors = np.array([
                                  (alpha, 1 - self.k_fold_results(**{
                                      pen_params.keys()[0]: alpha
                                  }).mean()) for alpha in bar.iter(values)
                                  ])
            fig, ax = plt.subplots()
            ax.plot(errors[:, 0], errors[:, 1])
            ax.set_title('{} test error rate'.format(self.method))
            ax.set_xlabel(
                'penalty parameter {}'.format(pen_params.keys()[0])
            )
            ax.set_ylabel('test error')
            plt.savefig('test_error_{METHOD}_{PARAM}.png'.format(
                METHOD=self.method,
                PARAM=pen_params.keys()[0]
            ))

        def _rforest_plot(self, pen_params):
            bar = Bar(
                width=40,
                suffix='%(percent)d%%'
            )

            X, Y = np.meshgrid(pen_params['n_estimators'],
                               pen_params['max_depth'])
            print 'Getting errors for {}...'.format(self.method)
            Z = np.array([
                             self.k_fold_results(**{
                                 'n_estimators': x,
                                 'max_depth': y
                             }).mean() for x in pen_params['n_estimators']
                             for y in bar.iter(pen_params['max_depth'])
                             ])
            Z.shape = (len(X), len(Y))
            fig, ax = plt.subplots()

            p = ax.contourf(X, Y, Z,
                            cmap='RdYlBu')
            ax.set_xlabel('n_estimators')
            ax.set_ylabel('max_depth')
            ax.set_title('rforest test error rate')
            plt.colorbar(p)
            plt.savefig('test_error_rforest.png')

        if self.method == 'rforest':
            _rforest_plot(self, pen_params)
        else:
            _non_forest_plot(self, pen_params)

    def raw_oos_pred(self):
        if not self.bestmod:
            raise ValueError('Model not yet trained. Run sp.tune_model()')

        ind = self.clean_X.index
        clean_oos_preds = pd.Series(
            self.bestmod.predict(self.clean_X),
            name='prediction',
            index=self.clean_X.index
        )

        print type(clean_oos_preds)
        print clean_oos_preds.head()

        return pd.concat(
            [
                self.data.loc[
                    ind,
                    ['t.ats_margin', 'real_spread', 'cover', 'o.team']
                ],
                clean_oos_preds
            ],
            axis=1
        )


    def oos_score(self):
        """
        Returns score of self.model on uncontaminated clean data.
        """

        if not self.bestmod:
            raise ValueError('Model not yet trained. Run sp.tune_model()')
        return self.bestmod.score(
            self.clean_X,
            self.clean_y
        )

    def new_predictions(self):
        if not self.bestmod:
            raise ValueError('Model not yet trained. Run sp.tune_model()')
        add_on = ' + C(otype):C(dtype)' if self.method != 'rforest' else \
            ''
        X = dmatrix(
            ' + '.join(self.predictors) + add_on,
            data=self.to_predict,
            return_type='dataframe'
        )
        return pd.DataFrame(
            {'predictions': self.bestmod.predict(X)},
            index=pd.MultiIndex.from_tuples(
                zip(self.to_predict.index.get_level_values('team'),
                    self.to_predict.index.get_level_values('date'),
                    self.to_predict['o.team']),
                names=['home team', 'date', 'away team']
            ),
        )
