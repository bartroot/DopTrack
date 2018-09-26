import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dtable
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
from plotly import tools
import time

from doptools.analysis import BulkAnalysis, ResidualAnalysis

data = BulkAnalysis().data
data_labels = {'tca': {'name': 'TCA - Time of closest approach', 'unit': ''},
               'fca': {'name': 'FCA - Frequency at closest approach', 'unit': '(Hz)'},
               'dtca': {'name': 'dTCA - Time error', 'unit': '(s)'}}


app = dash.Dash()
app.title = 'DopTrack'


app.css.append_css({"external_url": "https://www.tudelft.nl/typo3conf/ext/site_tud/Resources/Public/StyleSheets/dist.style.c1efa4e3be.min.css"})
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})


app.layout = html.Div(children=[

    html.Div(children=[
        html.Img(src='https://www.tudelft.nl/typo3conf/ext/tud_styling/Resources/Public/img/logo.svg',
                 style={'height': 55, 'align': 'left', 'margin': 10},
                 className='two columns'),

        html.H1(children='DopTrack Analysis Dashboard',
                style={'textAlign': 'right', 'margin': 10, 'color': 'white'},
                className='ten columns')],

        className='row', style={'background-color': '#00A6D6'}),

    html.Div(
        html.Div(className='row',
                 children=[
                    html.Div(className='six columns',
                        children=[

                        html.Div(className='five columns offset-by-one column',
                                 children=[
                                    dcc.Dropdown(
                                        id='ykey-dropdown',
                                        options=[{'label': val['name'], 'value': key}
                                                 for key, val in data_labels.items()],
                                        value='dtca'
                                        )],
                            style={'fontSize': 12, 'marginBottom': 30}
                            ),

                        html.Div(className='five columns',
                                 children=[
                                    dcc.Dropdown(
                                        id='xkey-dropdown',
                                        options=[{'label': val['name'], 'value': key}
                                                 for key, val in data_labels.items()],
                                        value='fca'
                                        )],
                            style={'fontSize': 12, 'marginBottom': 30}
                            ),

                        html.Div(className='twelve columns',
                                     children=[
                                             dcc.Graph(id='bulk-graph',
                                                       config={'displaylogo': False,
                                                               'modeBarButtonsToRemove': ['sendDataToCloud',
                                                                                          'select2d',
                                                                                          'lasso2d',
                                                                                          'autoScale2d',
                                                                                          'hoverClosestCartesian',
                                                                                          'hoverCompareCartesian',
                                                                                          'toggleSpikelines']})
                                ]),
                        ]
                    ),


                html.Div(className='six columns',
                         children=[

                            html.Div(className='three columns',
                                     children=[
                                             dcc.Checklist(
                                                id='checklist',
                                                options=[
                                                    {'label': 'Bad data   ', 'value': 'bad_data'},
                                                    {'label': 'Shutdown   ', 'value': 'shutdown'}
                                                ],
                                                values=[],
    #                                            labelStyle={'display': 'inline-block'},
                                                className='offset-by-four columns'
                                             )
                                ],
                                style={'margin': 0}),

                            html.Button('Previous',
                                        id='button_prev',
                                        className='two columns',
                                        n_clicks_timestamp='0',
                                        style={'marginBottom': 25}),

                            html.Button('Save',
                                        id='button_save',
                                        className='two columns',
                                        n_clicks_timestamp='0',
                                        style={'marginBottom': 25}),

                            html.Button('Next',
                                        id='button_next',
                                        className='two columns',
                                        n_clicks_timestamp='0',
                                        style={'marginBottom': 25}),

                            html.Div(className='twelve columns',
                                     children=[
                                             dcc.Graph(id='pass-graph',
                                                       config={'displaylogo': False,
                                                               'modeBarButtonsToRemove': ['sendDataToCloud',
                                                                                          'select2d',
                                                                                          'lasso2d',
                                                                                          'autoScale2d',
                                                                                          'hoverClosestCartesian',
                                                                                          'hoverCompareCartesian',
                                                                                          'toggleSpikelines']})
                                ]),

                            ]
                        ),

        ]), className='container', style={'width': '95%', 'max-width': 1920}),#, 'max-width': 50000})

#    dtable.DataTable(
#        # Initialise the rows
#        rows=data.to_dict('records'),
#        row_selectable=True,
#        filterable=True,
#        sortable=True,
#        selected_row_indices=[],
#        id='table'
#    ),

    html.Div(id='hidden-div', style={'display': 'none'})

])


@app.callback(
        Output('bulk-graph', 'figure'),
        [Input('xkey-dropdown', 'value'),
         Input('ykey-dropdown', 'value')])
def update_bulk_graph(xkey, ykey):
    xtitle = ' '.join(['<b>', data_labels[xkey]['name'], data_labels[xkey]['unit'], '</b>'])
    ytitle = ' '.join(['<b>', data_labels[ykey]['name'], data_labels[ykey]['unit'], '</b>'])
    return {
        'data': [
            go.Scatter(
                    x=data[data['timeofday'] == 'morning'][xkey],
                    y=data[data['timeofday'] == 'morning'][ykey],
                    text=data[data['timeofday'] == 'morning'].index,
                    mode='markers',
                    marker={
                        'size': 8,
                        'line': {'width': 1, 'color': 'white'}
                    },
                    name='Morning'
            ),
            go.Scatter(
                    x=data[data['timeofday'] == 'evening'][xkey],
                    y=data[data['timeofday'] == 'evening'][ykey],
                    text=data[data['timeofday'] == 'evening'].index,
                    mode='markers',
                    marker={
                        'size': 8,
                        'line': {'width': 1, 'color': 'white'}
                    },
                    name='Evening'
            )
        ],
        'layout': go.Layout(
#            title='Complete Dataset',
            xaxis={'title': xtitle, 'showline': True},
            yaxis={'title': ytitle, 'showline': True},
            hovermode='closest',
            legend={'x': 0, 'y': 1},
            height=715,
            margin={'t': 40, 'r': 40, 'b': 70, 'l': 100}
        )
    }




@app.callback(Output('pass-graph', 'figure'),
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
        print(i, dataid)
    else:
        dataid = clickData['points'][0]['text']
        i = int(np.where(data.index == dataid)[0])
        print(i, dataid)

    pass_data = ResidualAnalysis(dataid)

    fig = tools.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0)
    fig['layout']['legend'] = {'x': 0.03, 'y': 1, 'xanchor': 'left'}
    fig.append_trace({
        'x': pass_data.time,
        'y': pass_data.doptrack.rangerate,
        'name': 'DopTrack range rate',
        'mode': 'markers',
        'type': 'scatter'
    }, 1, 1)
    fig.append_trace({
        'x': pass_data.time,
        'y': pass_data.tle.rangerate,
        'name': 'TLE range rate',
        'mode': 'lines',
        'type': 'scatter'
    }, 1, 1)
    fig.append_trace({
        'x': pass_data.time,
        'y': pass_data.residual_first,
        'name': 'First residual',
        'mode': 'lines+markers',
        'type': 'scatter'
    }, 2, 1)
    fig['layout'].update(height=715,
                         title=f'<b>Satellite Pass {i+1} of {len(data)}: {dataid}</b>',
                         titlefont={'size': 14},
                         margin={'t': 50, 'r': 50, 'b': 70, 'l': 80})
    fig['layout']['xaxis1'].update(title='<b>Time UTC</b>', showline=True)
    fig['layout']['yaxis1'].update(title='<b>(m/s)</b>', showline=True)
    fig['layout']['yaxis2'].update(title='<b>(m/s)</b>', showline=True)
    return fig


@app.callback(Output('hidden-div', 'children'),
              [Input('button_save', 'n_clicks_timestamp')],
              [State('checklist', 'values')])
def update_passinfo(n_clicks, checklist):



    print('TTTTTT:', n_clicks)
    print('TTTTTT:', checklist)
    return None



if __name__ == '__main__':
    app.run_server(debug=False)
