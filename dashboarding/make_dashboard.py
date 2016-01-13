## TODO:
#  1. add season selection TapTool
#  2. add spread selection box select

import datetime as dt

import numpy as np
import pandas as pd
from bokeh.io import curdoc, gridplot
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.models import HoverTool, HBox, VBox, Slider, Select
from bokeh.models.callbacks import CustomJS
from bokeh.models.ranges import FactorRange, Range1d
from bokeh.plotting import Figure, output_file, show, ColumnDataSource

train_data = pd.read_csv('preds_20160110ins.csv').rename(
    columns={'t.ats_margin': 'ats_margin'}
)
train_data['type'] = 'train'
test_data = pd.read_csv('preds_20160110oos.csv').rename(
    columns={'t.ats_margin': 'ats_margin'}
)
test_data['type'] = 'test'

test_source = ColumnDataSource(
    {c: test_data[c] for c in test_data.columns}
)
train_source = ColumnDataSource(
    {c: test_data[c] for c in train_data.columns}
)

data = pd.concat([train_data, test_data], ignore_index=True)

dataset_select = Select(
    value='train',
    title='Select dataset:',
    options=['train', 'test']
)

def accuracy_for_group(group):
    return (
        (group['prediction'] == group['cover']).sum() /
        float(len(group))
    )

def accuracy_by(data, column):
    grouped = data.groupby(column)
    accuracy = grouped.apply(accuracy_for_group)
    accuracy.name = 'accuracy'
    count = grouped[['team']].count()
    out = count.join(accuracy).reset_index()
    out.columns = [column, 'count', 'accuracy']
    out['rect_centers'] = out['accuracy'] / 2
    return out


# set up three Figures
# note the capital F -- figure creates a Figure and adds it to document,
# which causes problems if you then try to use that figure in layouts.
# Figure just creates a Figure instance that you can then add to the document
# later. See this github issue:
# https://github.com/bokeh/bokeh/issues/3531
p1 = Figure(title='Accuracy by Season',
            plot_width=700, plot_height=300,
            x_range=Range1d(2002, 2015),
            y_range=Range1d(0,1),
            tools='hover,reset') # accuracy by season
p2 = Figure(title='Accuracy by Spread',
            plot_width=700, plot_height=300,
            x_range=Range1d(-40, 40), y_range=Range1d(0,1),
            tools='hover,reset') # accuracy by spread
p3 = Figure(title='Accuracy by Prediction',
            plot_width=400, plot_height=645,
            x_range=Range1d(-.5, 1.5), y_range=Range1d(0,1),
            tools='hover,reset') # pred cover vs. not

# set up ColumnDataSource and draw rectanges for p1
source1 = ColumnDataSource(data=accuracy_by(train_data, 'season'))
p1.rect(x='season', y='rect_centers', width=0.75, height='accuracy',
        color='#34a748', # something like picwell green
        alpha=0.8,
        source=source1)

# set up ColumnDataSource and draw rectanges for p2
source2 = ColumnDataSource(data=accuracy_by(train_data, 'ats_margin'))
p2.rect(x='ats_margin', y='rect_centers', width=0.375, height='accuracy',
        color='#fbb949', # something like picwell orange
        alpha=0.8,
        source=source2)

# set up ColumnDataSource and draw rectanges for p3
xtab3 = pd.crosstab(index=train_data['prediction'],
                    columns=train_data['cover'],
                    aggfunc=pd.Series.mean) / float(len(train_data))
data3 = pd.DataFrame([[0, xtab3.loc[0,0] / xtab3.loc[0].sum()],
                      [1, xtab3.loc[1,1] / xtab3.loc[1].sum()]],
                      columns=['prediction', 'accuracy'])
data3['rect_centers'] = data3['accuracy'] / 2.
source3 = ColumnDataSource(data=data3)
p3.rect(x='prediction', y='rect_centers', width=0.75, height='accuracy',
        color='#f06564', # something like picwell pink
        alpha=0.8,
        source=source3)

# add interaction for Select
def get_source_1(train_or_test, data=data):
    dataset = data[data['type'] == train_or_test].copy()
    source1.data = accuracy_by(dataset, 'season').to_dict(orient='list')

def get_source_2(train_or_test, data=data):
    dataset = data[data['type'] == train_or_test].copy()
    source2.data = accuracy_by(dataset, 'ats_margin').to_dict(orient='list')

def get_source_3(train_or_test, data=data):
    dataset = data[data['type'] == train_or_test].copy()
    xtab3 = pd.crosstab(index=dataset['prediction'],
                        columns=dataset['cover'],
                        aggfunc=pd.Series.mean) / float(len(train_data))
    data3 = pd.DataFrame([[0, xtab3.loc[0,0] / xtab3.loc[0].sum()],
                          [1, xtab3.loc[1,1] / xtab3.loc[1].sum()]],
                         columns=['prediction', 'accuracy'])
    data3['rect_centers'] = data3['accuracy'] / 2.
    source3.data = data3.to_dict(orient='list')

def update_sources(attrname, old, new): # magic parameters from JS
    get_source_1(new)
    get_source_2(new)
    get_source_3(new)

dataset_select.on_change('value', update_sources)

# add some tooltips for p1
hover1 = p1.select_one(HoverTool)
hover1.point_policy = 'snap_to_data'
hover1.tooltips = [
    ('Season', '@season'),
    ('Accuracy', '@accuracy'),
    ('Games', '@count')
]

# add some tooltips for p2
hover2 = p2.select_one(HoverTool)
hover2.point_policy = 'snap_to_data'
hover2.tooltips = [
    ('Spread', '@ats_margin'),
    ('Accuracy', '@accuracy'),
    ('Games', '@count')
]

# add some tooltips for p3
hover3 = p3.select_one(HoverTool)
hover3.point_policy = 'follow_mouse'
hover3.tooltips = [
    ('Prediction', '@prediction'),
    ('Accuracy', '@accuracy')
]

vlayout = VBox(p1, p2)
hlayout = HBox(vlayout, p3)
with open('dashboard.html', 'w') as outf:
    outf.write(file_html(hlayout, CDN, 'train data'))
layout = VBox(dataset_select, hlayout)
curdoc().add_root(layout)
