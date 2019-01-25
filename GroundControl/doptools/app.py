import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
from plotly import tools
import time
import pandas as pd
import uncertainties

from doptools.data import L1B
from doptools.analysis import BulkAnalysis, ResidualAnalysis
from doptools.config import Config


data = BulkAnalysis().data

data_labels = {'tca': {'name': 'TCA - Datetime of closest approach', 'unit': ''},
               'fca': {'name': 'FCA - Frequency at closest approach', 'unit': '(Hz)'},
               'dtca': {'name': 'dTCA - Time error', 'unit': '(s)'},
               'tca_time_plotly': {'name': 'TCA - Time of day of closest approach', 'unit': ''},
               'rmse': {'name': 'RMSE - Error between data and tanh fit', 'unit': '(Hz)'},
               'max_elevation': {'name': 'Maximum elevation', 'unit': '(deg)'}}
data['tca_time_plotly'] = [dt.replace(year=2000, month=1, day=1) for dt in data['tca']]


app = dash.Dash()
app.title = 'DopTrack'


app.css.append_css({"external_url": "https://www.tudelft.nl/typo3conf/ext/site_tud/Resources/Public/StyleSheets/dist.style.c1efa4e3be.min.css"})
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})


app.layout = html.Div(
        children=[
                html.Div(
                        children=[
                                html.Img(
                                        src='https://www.tudelft.nl/typo3conf/ext/tud_styling/Resources/Public/img/logo.svg',
                                        style={'height': 55, 'align': 'left', 'margin': 10},
                                        className='two columns'),

                                html.H1(
                                        children='DopTrack Analysis Dashboard',
                                        style={'textAlign': 'right', 'margin': 10, 'color': 'white'},
                                        className='ten columns')],

                        className='row', style={'background-color': '#00A6D6'}),

                html.Div(
                        html.Div(
                                className='row',
                                children=[
                                        html.Div(
                                                className='six columns',
                                                children=[
                                                        html.Div(
                                                                className='five columns offset-by-one column',
                                                                children=[
                                                                        dcc.Dropdown(
                                                                                id='ykey-dropdown',
                                                                                options=[
                                                                                        {'label': val['name'], 'value': key}
                                                                                        for key, val in data_labels.items()],
                                                                                value='dtca')],
                                                                style={'fontSize': 12, 'marginBottom': 30}),

                                                        html.Div(
                                                                className='five columns',
                                                                children=[
                                                                        dcc.Dropdown(
                                                                                id='xkey-dropdown',
                                                                                options=[
                                                                                        {'label': val['name'], 'value': key}
                                                                                        for key, val in data_labels.items()],
                                                                                value='tca')],
                                                                style={'fontSize': 12, 'marginBottom': 30}),

                                                        html.Div(
                                                                className='twelve columns',
                                                                children=[
                                                                        dcc.Graph(
                                                                                id='bulk-graph',
                                                                                config={
                                                                                        'displaylogo': False,
                                                                                        'modeBarButtonsToRemove': [
                                                                                                'sendDataToCloud',
                                                                                                'select2d',
                                                                                                'lasso2d',
                                                                                                'autoScale2d',
                                                                                                'hoverClosestCartesian',
                                                                                                'hoverCompareCartesian',
                                                                                                'toggleSpikelines']})])]),

                                        html.Div(
                                                className='six columns',
                                                children=[
                                                        html.Div(
                                                                className='three columns',
                                                                children=[
                                                                        dcc.Checklist(
                                                                                id='checklist',
                                                                                options=[
                                                                                        {'label': 'Bad data   ', 'value': 'bad_data'},
                                                                                        {'label': 'Reboot   ', 'value': 'reboot'}],
                                                                                values=[],
                                                                                #  labelStyle={'display': 'inline-block'},
                                                                                className='offset-by-four columns')],
                                                                style={'margin': 0}),

                                                        html.Button(
                                                                'Previous',
                                                                id='button_prev',
                                                                className='two columns',
                                                                n_clicks_timestamp='0',
                                                                style={'marginBottom': 25}),

                                                        html.Button(
                                                                'Save',
                                                                id='button_save',
                                                                className='two columns',
                                                                n_clicks_timestamp='0',
                                                                style={'marginBottom': 25}),

                                                        html.Button(
                                                                'Next',
                                                                id='button_next',
                                                                className='two columns',
                                                                n_clicks_timestamp='0',
                                                                style={'marginBottom': 25}),

                                                        html.Div(
                                                                className='twelve columns',
                                                                children=[
                                                                        dcc.Graph(
                                                                                id='pass-graph',
                                                                                config={
                                                                                        'displaylogo': False,
                                                                                        'modeBarButtonsToRemove': [
                                                                                                'sendDataToCloud',
                                                                                                'select2d',
                                                                                                'lasso2d',
                                                                                                'autoScale2d',
                                                                                                'hoverClosestCartesian',
                                                                                                'hoverCompareCartesian',
                                                                                                'toggleSpikelines']})])])]),
                        className='container',
                        style={
                                'width': '95%',
                                'max-width': 1920,
                                'margin': 10}),

                html.Div(id='hidden-div', style={'display': 'none'})])


@app.callback(
        Output('bulk-graph', 'figure'),
        [Input('xkey-dropdown', 'value'), Input('ykey-dropdown', 'value')])
def update_bulk_graph(xkey, ykey):
    xtitle = ' '.join(['<b>', data_labels[xkey]['name'], data_labels[xkey]['unit'], '</b>'])
    ytitle = ' '.join(['<b>', data_labels[ykey]['name'], data_labels[ykey]['unit'], '</b>'])

    plot = dict()
    plot['data'] = []
    for timeofday in ['morning', 'evening']:

        if xkey == 'fca' and ykey == 'fca':

            plot['data'].append(
                go.Scatter(
                        x=[uncertainties.ufloat_fromstr(val).n for val in data[data['timeofday'] == timeofday][xkey]],
                        error_x=dict(
                                type='data',
                                array=[uncertainties.ufloat_fromstr(val).s for val in data[data['timeofday'] == timeofday][xkey]],
                                visible=True),
                        y=[uncertainties.ufloat_fromstr(val).n for val in data[data['timeofday'] == timeofday][ykey]],
                        error_y=dict(
                                type='data',
                                array=[uncertainties.ufloat_fromstr(val).s for val in data[data['timeofday'] == timeofday][ykey]],
                                visible=True),
                        text=data[data['timeofday'] == timeofday].index,
                        mode='markers',
                        marker={
                            'size': 8,
                            'line': {'width': 1, 'color': 'white'}},
                        name=timeofday))
        elif xkey == 'fca':
            plot['data'].append(
                go.Scatter(
                        x=[uncertainties.ufloat_fromstr(val).n for val in data[data['timeofday'] == timeofday][xkey]],
                        error_x=dict(
                                type='data',
                                array=[uncertainties.ufloat_fromstr(val).s for val in data[data['timeofday'] == timeofday][xkey]],
                                visible=True),
                        y=data[data['timeofday'] == timeofday][ykey],
                        text=data[data['timeofday'] == timeofday].index,
                        mode='markers',
                        marker={
                            'size': 8,
                            'line': {'width': 1, 'color': 'white'}},
                        name=timeofday))
        elif ykey == 'fca':
            plot['data'].append(
                go.Scatter(
                        x=data[data['timeofday'] == timeofday][xkey],
                        y=[uncertainties.ufloat_fromstr(val).n for val in data[data['timeofday'] == timeofday][ykey]],
                        error_y=dict(
                                type='data',
                                array=[uncertainties.ufloat_fromstr(val).s for val in data[data['timeofday'] == timeofday][ykey]],
                                visible=True),
                        text=data[data['timeofday'] == timeofday].index,
                        mode='markers',
                        marker={
                            'size': 8,
                            'line': {'width': 1, 'color': 'white'}},
                        name=timeofday))
        else:
            plot['data'].append(
                go.Scatter(
                        x=data[data['timeofday'] == timeofday][xkey],
                        y=data[data['timeofday'] == timeofday][ykey],
                        text=data[data['timeofday'] == timeofday].index,
                        mode='markers',
                        marker={
                            'size': 8,
                            'line': {'width': 1, 'color': 'white'}},
                        name=timeofday))

    plot['layout'] = go.Layout(
            #  title='Complete Dataset',
            xaxis={'title': xtitle, 'showline': True},
            yaxis={'title': ytitle, 'showline': True},
            hovermode='closest',
            legend={'x': 0, 'y': 1},
            height=715,
            margin={'t': 40, 'r': 40, 'b': 70, 'l': 100})

    return plot


@app.callback(
        Output('pass-graph', 'figure'),
        [Input('bulk-graph', 'clickData'),
         Input('button_prev', 'n_clicks_timestamp'),
         Input('button_next', 'n_clicks_timestamp')],
        [State('pass-graph', 'figure')])
def update_pass_graphs_from_bulk(clickData, t_prev, t_next, figure):

    t_prev, t_next = int(t_prev)/1000, int(t_next)/1000

    if clickData is None and (t_prev == 0 and t_next == 0):
        i = 0
        dataid = data.index[0]
    elif (time.time() - int(t_prev)) < 1 or (time.time() - int(t_next)) < 1:
        dataid = figure['layout']['title'][-31:-4]
        i = int(np.where(data.index == dataid)[0])
        if t_prev > t_next:
            if i == 0:
                dataid = dataid
            else:
                dataid = data.index[i - 1]
        else:
            if i == len(data) - 1:
                dataid = dataid
            else:
                dataid = data.index[i + 1]
        i = int(np.where(data.index == dataid)[0])
    else:
        dataid = clickData['points'][0]['text']
        i = int(np.where(data.index == dataid)[0])

    pass_data = ResidualAnalysis(L1B.load(dataid))

    fig = tools.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0)
    fig['layout']['legend'] = {'x': 0.03, 'y': 1, 'xanchor': 'left'}

    fig.append_trace(
            {'x': pass_data.time,
             'y': pass_data.dataL1C.rangerate,
             'name': 'DopTrack range rate',
             'mode': 'markers',
             'type': 'scatter'},
            1,
            1)

    fig.append_trace(
            {'x': pass_data.time,
             'y': pass_data.dataTLE.rangerate,
             'name': 'TLE range rate',
             'mode': 'lines',
             'type': 'scatter'},
            1,
            1)

    fig.append_trace(
            {'x': pass_data.time,
             'y': pass_data.first_residual,
             'name': 'First residual',
             'mode': 'lines+markers',
             'type': 'scatter'},
            2,
            1)

    fig['layout'].update(
            height=715,
            title=f'<b>Satellite Pass {i+1} of {len(data)}: {dataid}</b>',
            titlefont={'size': 14},
            margin={'t': 50, 'r': 50, 'b': 70, 'l': 80})

    fig['layout']['xaxis1'].update(title='<b>Time UTC</b>', showline=True)
    fig['layout']['yaxis1'].update(title='<b>(m/s)</b>', showline=True)
    fig['layout']['yaxis2'].update(title='<b>(m/s)</b>', showline=True)

    return fig


@app.callback(
        Output('checklist', 'values'),
        [Input('bulk-graph', 'clickData'),
         Input('button_prev', 'n_clicks_timestamp'),
         Input('button_next', 'n_clicks_timestamp')],
        [State('pass-graph', 'figure')])
def update_feature_checklist(clickData, t_prev, t_next, figure):

    t_prev, t_next = int(t_prev)/1000, int(t_next)/1000

    if clickData is None and (t_prev == 0 and t_next == 0):
        i = 0
        dataid = data.index[0]
    elif (time.time() - int(t_prev)) < 1 or (time.time() - int(t_next)) < 1:
        dataid = figure['layout']['title'][-31:-4]
        i = int(np.where(data.index == dataid)[0])
        if t_prev > t_next:
            if i == 0:
                dataid = dataid
            else:
                dataid = data.index[i - 1]
        else:
            if i == len(data) - 1:
                dataid = dataid
            else:
                dataid = data.index[i + 1]
        i = int(np.where(data.index == dataid)[0])
    else:
        dataid = clickData['points'][0]['text']
        i = int(np.where(data.index == dataid)[0])

    filepath = Config().paths['default'] / 'pass_features.csv'
    if not filepath.is_file():
        return []
    features = pd.read_csv(filepath)
    features.set_index('dataid', inplace=True)
    if dataid not in features.index:
        return []

    feature_list = [feature for feature, boolean in features.loc[dataid].iteritems() if boolean]

    return feature_list


@app.callback(
        Output('hidden-div', 'children'),
        [Input('button_save', 'n_clicks_timestamp')],
        [State('checklist', 'values'),
         State('pass-graph', 'figure')])
def update_pass_features(n_clicks, checklist, figure):

    if figure is None:
        return None
    filepath = Config().paths['default'] / 'pass_features.csv'
    if filepath.is_file():
        features = pd.read_csv(filepath)
        new_dataids = set.difference(set(data.index), set(features['dataid']))
        new_features = pd.DataFrame([[new_dataid, False, False] for new_dataid in new_dataids],
                                    columns=['dataid', 'bad_data', 'reboot'])
        features = features.append(new_features)
        # TODO fix bug where new passes in data are not added to info
    else:
        features = pd.DataFrame({'dataid': data.index,
                                 'bad_data': [False] * len(data.index),
                                 'reboot': [False] * len(data.index)
                                 })
    features.set_index('dataid', inplace=True)

    dataid = figure['layout']['title'][-31:-4]

    features.loc[dataid, 'bad_data'] = True if 'bad_data' in checklist else False
    features.loc[dataid, 'reboot'] = True if 'reboot' in checklist else False

    features.sort_index(inplace=True)
    features.to_csv(filepath)

    return None


if __name__ == '__main__':
    app.run_server(debug=False)
