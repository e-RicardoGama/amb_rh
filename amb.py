import dash_bootstrap_components
import pandas as pd
import plotly
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import openpyxl
from dash.exceptions import PreventUpdate
from dash_bootstrap_templates import load_figure_template

load_figure_template('superhero')

# import from folders/theme changer
from app import *
# ========= Autenticação ======= #

# ========== Styles ============ #

tab_card = {'height': '100%'}

main_config = {
    "hovermode": "x unified",
    "legend": {"yanchor":"top",
                "y":0.9,
                "xanchor":"left",
                "x":0.1,
                "title": {"text": None},
                "font" :{"color":"white"},
                "bgcolor": "rgba(0,0,0,0.5)"},
    "margin": {"l":10, "r":10, "t":10, "b":10}
}

config_graph = {"displayModeBar": False, "showTips": False}

#=========== Jupyter ==============

df = pd.read_csv('datasets/base_seg2.csv',index_col=0)

df['ID_Func'] = df['ID_Func'].astype(str)
df['Emp'] = 'E1'

df_orig = df.copy()

perfil = {'Perfil 1': 1, 'Perfil 2': 2, 'Perfil 3': 3, 'Perfil 4': 4}

df['Perfil'] = df['Perfil'].map(perfil)

# Criando opções pros filtros que virão
options_month = [{'label': 'Todos Perfis', 'value': 0}]
for i, j in zip(df_orig['Perfil'].unique(), df['Perfil'].unique()):
    options_month.append({'label': i, 'value': j})
options_month = sorted(options_month, key=lambda x: x['value'])

#=========== Funções ============

def tot_depto(df):
    if df['Departamento'] == 'Pesq_Desenv':
        return 961
    elif df['Departamento'] == 'Vendas':
        return 446
    else:
        return 63

def tot_grau(df):
    if df['Grau Instrução'] == 'Pos_Graduado':
        return 572
    elif df['Grau Instrução'] == 'Mestrado':
        return 398
    elif df['Grau Instrução'] == 'Faculdade':
        return 282
    elif df['Grau Instrução'] == 'Ensino_Mèdio':
        return 170
    else:
        return 48

def tot_gen(df):
    if df['Gênero'] == 'Masculino':
        return 882
    else:
        return 588

def tot_est(df):
    if df['Estado Civil'] == 'Casado':
        return 673
    elif df['Estado Civil'] == 'Solteiro':
        return 470
    else:
        return 327

def perfil_filtro(perfil):
    if perfil == 0:
        mask = df['Perfil'].isin(df['Perfil'].unique())
    else:
        mask = df['Perfil'].isin([perfil])
    return mask

# =========  Layout  =========== #
app.layout = dbc.Container(children=[

    # Layout

    # Linha 1
    dbc.Row([
        dbc.Col([
            dbc.Col([
                html.Img(src=r'assets/logo.png',style={'height': '90%','width': '90%',
                                                               'margin-top':'1px'})
                    ]),
            ],sm=3, md=2,lg=1),
        dbc.Col([
            dbc.Card([
               dbc.CardBody([
                    dbc.Col([
                        html.H3('Clima Organizacional'),
                    ])
                ], style=tab_card)
            ])
        ],sm=12, md=10,lg=11),
    ], className='g-1 my-auto', style={'margin-top': '7px'}),

    # Linha 2
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6('Selecione o Perfil'),
                            ]),
                        ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                options=options_month,
                                value=0,
                                id='dropdown',
                                style={'margin-top':'10px'},
                                ),
                                ]),
                            ])
                ])
            ], style=tab_card)
        ], sm=4, md=4, lg=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6('Relatório dos Perfis para Download'),
                            ],sm=12,md=12,lg=12),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button('Todos Perfis', color='primary', id='btn_todos', n_clicks=0,
                                           style={'margin-top': '10px'}),
                                dcc.Download(id='todos'),
                                    ],sm=2,md=2,lg=3),
                            dbc.Col([
                                dbc.Button('Perfil 1', color='primary', id='btn_p1', n_clicks=0,
                                           style={'margin-top': '10px'}),
                                dcc.Download(id='p1'),
                                    ],sm=2,md=2,lg=3),
                            dbc.Col([
                                dbc.Button('Perfil 2', color='primary', id='btn_p2', n_clicks=0,
                                           style={'margin-top': '10px'}),
                                dcc.Download(id='p2'),
                                    ],sm=2,md=2,lg=3),
                            dbc.Col([
                                dbc.Button('Perfil 3', color='primary', id='btn_p3', n_clicks=0,
                                           style={'margin-top': '10px'}),
                                dcc.Download(id='p3'),
                                    ],sm=2,md=2,lg=3),
                            ]),
                        ]),
                ]),
            ], style=tab_card),
        ],sm=8,md=8,lg=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6('Médias por Perfil'),
                            ]),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id='graph7',className='dbc',config=config_graph),
                            ],sm=12,md=12,lg=6, style={'margin-top':'10px'}),
                            dbc.Col([
                                dcc.Graph(id='graph8',className='dbc',config=config_graph),
                            ],sm=12,md=12,lg=6, style={'margin-top':'10px'}),
                        ]),
                ]),
            ],style=tab_card)
        ],sm=6,md=6,lg=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6('Satisfação com Ambiente'),
                        ]),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='graph1', className='dbc', config=config_graph),
                        ], sm=12, md=6, lg=6),
                        dbc.Col([
                            dcc.Graph(id='graph12', className='dbc', config=config_graph),
                        ], sm=12, md=6, lg=6),
                    ]),
                ]),
            ],style=tab_card),
        ],sm=6,md=6,lg=4),
    ], className='g-1 my-auto', style={'margin-top': '7px'}),

    # Linha 3

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                html.H6('Envolvimento'),
                            ]),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id='graph2', className='dbc', config=config_graph),
                            ]),
                            dbc.Col([
                                dcc.Graph(id='graph13', className='dbc', config=config_graph),
                            ]),
                            ]),
                        ]),
                    ]),
                ],style=tab_card),
            ],sm=6,md=6,lg=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                html.H6('Satisfação com Trabalho'),
                            ]),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id='graph3', className='dbc', config=config_graph),
                            ]),
                            dbc.Col([
                                dcc.Graph(id='graph14', className='dbc', config=config_graph),
                            ]),
                            ]),
                        ]),
                    ]),
                ],style=tab_card),
            ],sm=6,md=6,lg=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                html.H6('Relacionamento'),
                            ]),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id='graph4', className='dbc', config=config_graph),
                            ]),
                            dbc.Col([
                                dcc.Graph(id='graph15', className='dbc', config=config_graph),
                            ]),
                            ]),
                        ]),
                    ]),
                ],style=tab_card),
            ],sm=6,md=6,lg=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                html.H6('Balanço Vida-Profissão'),
                            ]),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id='graph5', className='dbc', config=config_graph),
                            ]),
                            dbc.Col([
                                dcc.Graph(id='graph16', className='dbc', config=config_graph),
                            ]),
                            ]),
                        ]),
                    ]),
                ],style=tab_card),
            ],sm=6,md=6,lg=3),
    ], className='g-1 my-auto', style={'margin-top': '7px'}),

    # Linha 4

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                html.H6('Departamento')
                                ])
                            ]),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id='graph6', className='dbc', config=config_graph)
                            ]),
                        ])
                    ]),
                ]),
            ], style=tab_card),
        ],sm=12,md=12,lg=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                html.H6('Grau de Instrução')
                            ])
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id='graph9',className='dbc',config=config_graph)
                            ])
                        ]),
                    ])
                ])
            ], style=tab_card)
            ],sm=12,md=12,lg=6)
    ], className='g-1 my-auto', style={'margin-top': '7px'}),

    # Linha 5

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                html.H6('Gênero')
                            ])
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id='graph10', className='dbc', config=config_graph)
                            ]),
                        ])
                    ]),
                ]),
            ], style=tab_card),
        ], sm=12, md=12, lg=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                html.H6('Estado Civil')
                            ])
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id='graph11', className='dbc', config=config_graph)
                            ])
                        ]),
                    ])
                ])
            ], style=tab_card)
        ], sm=12, md=12, lg=6)
    ], className='g-1 my-auto', style={'margin-top': '7px'}),

], fluid=True, style={'height': '100vh'})

#========= CallBack =========

# Download Perfis
# Todos Perfis

@app.callback(
    Output('todos', 'data'),
    Input('btn_todos', 'n_clicks'),
    prevent_initial_call=True,
)

def func(n_clicks):

    if n_clicks == None:
        raise PreventUpdate

    tp = df
    tp = tp.sort_values('Perfil')

    return dcc.send_data_frame(tp.to_excel, "Todos Perfis.xlsx", sheet_name="Sheet_name_1")

# Perfil 1

@app.callback(
    Output('p1', 'data'),
    Input('btn_p1', 'n_clicks'),
    prevent_initial_call=True,
)

def func(n_clicks):

    if n_clicks == None:
        raise PreventUpdate

    p1 = df.loc[df['Perfil'] == 1]

    return dcc.send_data_frame(p1.to_excel, "Perfil 1.xlsx", sheet_name="Sheet_name_1")

# Perfil 2

@app.callback(
    Output('p2', 'data'),
    Input('btn_p2', 'n_clicks'),
    prevent_initial_call=True,
)

def func(n_clicks):

    if n_clicks == None:
        raise PreventUpdate

    p2 = df.loc[df['Perfil'] == 2]

    return dcc.send_data_frame(p2.to_excel, "Perfil 2.xlsx", sheet_name="Sheet_name_1")

# Perfil 3

@app.callback(
    Output('p3', 'data'),
    Input('btn_p3', 'n_clicks'),
    prevent_initial_call=True,
)

def func(n_clicks):

    if n_clicks == None:
        raise PreventUpdate

    p3 = df.loc[df['Perfil'] == 3]

    return dcc.send_data_frame(p3.to_excel, "Perfil 3.xlsx", sheet_name="Sheet_name_1")

# Graph 1

@app.callback(
    Output('graph1','figure'),
    Input('dropdown','value'),
)

def graph1(perfil):

    mask = perfil_filtro(perfil)
    sa = df.loc[mask]

    sa = sa['Satisfação Ambiente'].value_counts()
    sa = pd.DataFrame(sa).reset_index()
    sa.rename(columns={'index': 'Avaliação'}, inplace=True)

    fig1 = go.Figure()
    fig1.add_trace(go.Pie(labels=sa['Avaliação'], values=sa['Satisfação Ambiente'], hole=.5))
    fig1.update(layout_showlegend=False)
    fig1.update_layout(main_config, height=100, template='superhero')

    return fig1

# Graph 2

@app.callback(
    Output('graph2','figure'),
    Input('dropdown','value'),
)

def graph2(perfil):

    mask = perfil_filtro(perfil)
    env = df.loc[mask]

    env = env['Envolvimento'].value_counts()
    env = pd.DataFrame(env).reset_index()
    env.rename(columns={'index': 'Avaliação'}, inplace=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Pie(labels=env['Avaliação'], values=env['Envolvimento'], hole=.5))
    fig2.update(layout_showlegend=False)
    fig2.update_layout(main_config, height=120, template='superhero')

    return fig2

# Graph 3

@app.callback(
    Output('graph3','figure'),
    Input('dropdown','value'),
)

def graph3(perfil):

    mask = perfil_filtro(perfil)
    st = df.loc[mask]

    st = st['Satisfação Trabalho'].value_counts()
    st = pd.DataFrame(st).reset_index()
    st.rename(columns={'index': 'Avaliação'}, inplace=True)

    fig3 = go.Figure()
    fig3.add_trace(go.Pie(labels=st['Avaliação'], values=st['Satisfação Trabalho'], hole=.5))
    fig3.update(layout_showlegend=False)
    fig3.update_layout(main_config, height=120, template='superhero')

    return fig3

# Graph 4

@app.callback(
    Output('graph4','figure'),
    Input('dropdown','value'),
)

def graph4(perfil):

    mask = perfil_filtro(perfil)
    re = df.loc[mask]

    re = re['Relacionamento'].value_counts()
    re = pd.DataFrame(re).reset_index()
    re.rename(columns={'index': 'Avaliação'}, inplace=True)

    fig4 = go.Figure()
    fig4.add_trace(go.Pie(labels=re['Avaliação'], values=re['Relacionamento'], hole=.5))
    fig4.update(layout_showlegend=False)
    fig4.update_layout(main_config, height=120, template='superhero')

    return fig4

# Graph 5

@app.callback(
    Output('graph5','figure'),
    Input('dropdown','value'),
)

def graph5(perfil):

    mask = perfil_filtro(perfil)
    vp = df.loc[mask]

    vp = vp['Balanço Vida-Profissão'].value_counts()
    vp = pd.DataFrame(vp).reset_index()
    vp.rename(columns={'index': 'Avaliação'}, inplace=True)

    fig5 = go.Figure()
    fig5.add_trace(go.Pie(labels=vp['Avaliação'], values=vp['Balanço Vida-Profissão'], hole=.5))
    fig5.update(layout_showlegend=False)
    fig5.update_layout(main_config, height=120, template='superhero')

    return fig5

# Graph 6

@app.callback(
    Output('graph6','figure'),
    Input('dropdown','value'),
)

def graph6(perfil):

    mask = perfil_filtro(perfil)
    depto = df.loc[mask]

    depto = depto.groupby(['Departamento'])['ID_Func'].count().reset_index()
    depto['Total Func'] = depto.apply(tot_depto, axis=1)
    depto['% Func'] = (depto['ID_Func'] / depto['Total Func'] * 100).round(2)

    fig6 = ff.create_table(depto, height_constant=60)

    dados = depto['Departamento']
    func = depto['ID_Func']
    total_func = depto['Total Func']

    trace1 = go.Bar(x=dados, y=func,
                    marker=dict(color='#0099ff'),
                    name='Func',
                    xaxis='x2', yaxis='y2')
    trace2 = go.Bar(x=dados, y=total_func,
                    marker=dict(color='#404040'),
                    name='Total Func',
                    xaxis='x2', yaxis='y2')

    fig6.add_traces([trace1, trace2])

    # initialize xaxis2 and yaxis2
    fig6['layout']['xaxis2'] = {}
    fig6['layout']['yaxis2'] = {}

    # Edit layout for subplots
    fig6.layout.xaxis.update({'domain': [0, .65]})
    fig6.layout.xaxis2.update({'domain': [0.73, 1.]})

    # The graph's yaxis MUST BE anchored to the graph's xaxis
    fig6.layout.yaxis2.update({'anchor': 'x2'})

    # Update the margins to add a title and see graph x-labels.
    fig6.layout.margin.update({'t': 150, 'b': 100})
    fig6.update(layout_showlegend=False)
    fig6.update_layout(main_config, height=200, template='superhero')

    return fig6

# Graph 7

@app.callback(
    Output('graph7','figure'),
    Input('dropdown','value'),
)

def graph7(perfil):

    mask = perfil_filtro(perfil)
    renda = df.loc[mask]

    renda = renda.groupby('Emp')['Renda'].mean().reset_index().round(2)

    fig7 = go.Figure()
    fig7.add_trace(go.Indicator(mode='number',
                                 title={"text": f"<span style='font-size:80%'>Renda Média</span><br>"},
                                 value=renda['Renda'].iloc[0],
                                 number={'prefix': "$"},
                                 ))
    fig7.update_layout(margin=dict(l=0, r=0, t=20, b=20), height=100, template='superhero')

    return fig7

# Graph 8

@app.callback(
    Output('graph8','figure'),
    Input('dropdown','value'),
)

def graph8(perfil):

    mask = perfil_filtro(perfil)
    idade = df.loc[mask]

    idade = idade.groupby('Emp')['Idade'].mean().reset_index().round(2)

    fig8 = go.Figure()
    fig8.add_trace(go.Indicator(mode='number',
                                 title={"text": f"<span style='font-size:80%'>Média de Idade</span><br>"},
                                 value=idade['Idade'].iloc[0],
                                 number={'prefix': ""},
                                 ))
    fig8.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=100, template='superhero')

    return fig8

# Graph 9

@app.callback(
    Output('graph9','figure'),
    Input('dropdown','value'),
)

def graph9(perfil):

    mask = perfil_filtro(perfil)
    grau = df.loc[mask]

    grau = grau.groupby(['Grau Instrução'])['ID_Func'].count().reset_index()
    grau['Total Func'] = grau.apply(tot_grau, axis=1)
    grau['% Func'] = (grau['ID_Func'] / grau['Total Func'] * 100).round(2)

    fig9 = ff.create_table(grau, height_constant=60)

    dados = grau['Grau Instrução']
    func = grau['ID_Func']
    total_func = grau['Total Func']

    trace1 = go.Bar(x=dados, y=func,
                    marker=dict(color='#0099ff'),
                    name='Func',
                    xaxis='x2', yaxis='y2')
    trace2 = go.Bar(x=dados, y=total_func,
                    marker=dict(color='#404040'),
                    name='Total Func',
                    xaxis='x2', yaxis='y2')

    fig9.add_traces([trace1, trace2])

    # initialize xaxis2 and yaxis2
    fig9['layout']['xaxis2'] = {}
    fig9['layout']['yaxis2'] = {}

    # Edit layout for subplots
    fig9.layout.xaxis.update({'domain': [0, .65]})
    fig9.layout.xaxis2.update({'domain': [0.73, 1.]})

    # The graph's yaxis MUST BE anchored to the graph's xaxis
    fig9.layout.yaxis2.update({'anchor': 'x2'})

    # Update the margins to add a title and see graph x-labels.
    fig9.layout.margin.update({'t': 150, 'b': 100})
    fig9.update(layout_showlegend=False)
    fig9.update_layout(main_config, height=200, template='superhero')

    return fig9

# Graph 10

@app.callback(
    Output('graph10','figure'),
    Input('dropdown','value'),
)

def graph10(perfil):

    mask = perfil_filtro(perfil)
    gen = df.loc[mask]

    gen = gen.groupby(['Gênero'])['ID_Func'].count().reset_index()
    gen['Total Func'] = gen.apply(tot_gen, axis=1)
    gen['% Func'] = (gen['ID_Func'] / gen['Total Func'] * 100).round(2)

    fig10 = ff.create_table(gen, height_constant=60)

    dados = gen['Gênero']
    func = gen['ID_Func']
    total_func = gen['Total Func']

    trace1 = go.Bar(x=dados, y=func,
                    marker=dict(color='#0099ff'),
                    name='Func',
                    xaxis='x2', yaxis='y2')
    trace2 = go.Bar(x=dados, y=total_func,
                    marker=dict(color='#404040'),
                    name='Total Func',
                    xaxis='x2', yaxis='y2')

    fig10.add_traces([trace1, trace2])

    # initialize xaxis2 and yaxis2
    fig10['layout']['xaxis2'] = {}
    fig10['layout']['yaxis2'] = {}

    # Edit layout for subplots
    fig10.layout.xaxis.update({'domain': [0, .65]})
    fig10.layout.xaxis2.update({'domain': [0.73, 1.]})

    # The graph's yaxis MUST BE anchored to the graph's xaxis
    fig10.layout.yaxis2.update({'anchor': 'x2'})
    # fig10.layout.yaxis2.update({'title': 'Conflito'})

    # Update the margins to add a title and see graph x-labels.
    fig10.layout.margin.update({'t': 50, 'b': 100})
    fig10.update(layout_showlegend=False)
    fig10.update_layout(main_config, height=200, template='superhero')

    return fig10

# Graph 11
@app.callback(
    Output('graph11','figure'),
    Input('dropdown','value'),
)

def graph11(perfil):

    mask = perfil_filtro(perfil)
    ec = df.loc[mask]

    ec = ec.groupby(['Estado Civil'])['ID_Func'].count().reset_index()
    ec['Total Func'] = ec.apply(tot_est, axis=1)
    ec['% Func'] = (ec['ID_Func'] / ec['Total Func'] * 100).round(2)

    fig11 = ff.create_table(ec, height_constant=60)

    dados = ec['Estado Civil']
    func = ec['ID_Func']
    total_func = ec['Total Func']

    trace1 = go.Bar(x=dados, y=func,
                    marker=dict(color='#0099ff'),
                    name='Func',
                    xaxis='x2', yaxis='y2')
    trace2 = go.Bar(x=dados, y=total_func,
                    marker=dict(color='#404040'),
                    name='Total Func',
                    xaxis='x2', yaxis='y2')

    fig11.add_traces([trace1, trace2])

    # initialize xaxis2 and yaxis2
    fig11['layout']['xaxis2'] = {}
    fig11['layout']['yaxis2'] = {}

    # Edit layout for subplots
    fig11.layout.xaxis.update({'domain': [0, .65]})
    fig11.layout.xaxis2.update({'domain': [0.73, 1.]})

    # The graph's yaxis MUST BE anchored to the graph's xaxis
    fig11.layout.yaxis2.update({'anchor': 'x2'})
    # fig11.layout.yaxis2.update({'title': 'Conflito'})

    # Update the margins to add a title and see graph x-labels.
    fig11.layout.margin.update({'t': 50, 'b': 100})
    fig11.update(layout_showlegend=False)
    fig11.update_layout(main_config, height=200, template='superhero')

    return fig11

# Graph 12
@app.callback(
    Output('graph12','figure'),
    Input('dropdown','value'),
)

def graph12(perfil):

    mask = perfil_filtro(perfil)
    sa = df.loc[mask]

    sa = sa.groupby('Satisfação Ambiente')['ID_Func'].count().reset_index()
    sa = sa.sort_values('ID_Func',ascending=True)

    fig12 = go.Figure(go.Bar(
        x=sa['ID_Func'],
        y=sa['Satisfação Ambiente'],
        orientation='h',
        textposition='auto',
        text=sa['ID_Func'],
        insidetextfont=dict(family='Times', size=12)))
    fig12.update_layout(main_config, height=100, template='superhero')

    return fig12

# Graph 13
@app.callback(
    Output('graph13','figure'),
    Input('dropdown','value'),
)

def graph13(perfil):

    mask = perfil_filtro(perfil)
    env = df.loc[mask]

    env = env.groupby('Envolvimento')['ID_Func'].count().reset_index()
    env = env.sort_values('ID_Func',ascending=True)

    fig13 = go.Figure(go.Bar(
        x=env['ID_Func'],
        y=env['Envolvimento'],
        orientation='h',
        textposition='auto',
        text=env['ID_Func'],
        insidetextfont=dict(family='Times', size=12)))
    fig13.update_layout(main_config, height=120, template='superhero')

    return fig13

# Graph 14
@app.callback(
    Output('graph14','figure'),
    Input('dropdown','value'),
)

def graph14(perfil):

    mask = perfil_filtro(perfil)
    st = df.loc[mask]

    st = st.groupby('Satisfação Trabalho')['ID_Func'].count().reset_index()
    st = st.sort_values('ID_Func',ascending=True)

    fig14 = go.Figure(go.Bar(
        x=st['ID_Func'],
        y=st['Satisfação Trabalho'],
        orientation='h',
        textposition='auto',
        text=st['ID_Func'],
        insidetextfont=dict(family='Times', size=12)))
    fig14.update_layout(main_config, height=120, template='superhero')

    return fig14

# Graph 15
@app.callback(
    Output('graph15','figure'),
    Input('dropdown','value'),
)

def graph15(perfil):

    mask = perfil_filtro(perfil)
    rel = df.loc[mask]

    rel = rel.groupby('Relacionamento')['ID_Func'].count().reset_index()
    rel = rel.sort_values('ID_Func',ascending=True)

    fig15 = go.Figure(go.Bar(
        x=rel['ID_Func'],
        y=rel['Relacionamento'],
        orientation='h',
        textposition='auto',
        text=rel['ID_Func'],
        insidetextfont=dict(family='Times', size=12)))
    fig15.update_layout(main_config, height=120, template='superhero')

    return fig15

# Graph 16
@app.callback(
    Output('graph16','figure'),
    Input('dropdown','value'),
)

def graph16(perfil):

    mask = perfil_filtro(perfil)
    bvp = df.loc[mask]

    bvp = bvp.groupby('Balanço Vida-Profissão')['ID_Func'].count().reset_index()
    bvp = bvp.sort_values('ID_Func',ascending=True)

    fig16 = go.Figure(go.Bar(
        x=bvp['ID_Func'],
        y=bvp['Balanço Vida-Profissão'],
        orientation='h',
        textposition='auto',
        text=bvp['ID_Func'],
        insidetextfont=dict(family='Times', size=12)))
    fig16.update_layout(main_config, height=120, template='superhero')

    return fig16


# Run server
if __name__ == '__main__':
    app.run_server(threaded=True)