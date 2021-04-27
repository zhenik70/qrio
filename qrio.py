#!/usr/bin/env python
# coding: utf-8

# In[30]:




import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as table
from dash.dependencies import Output, Input
import dash_daq as daq

#from pylab import mpl, plt
#import cufflinks as cf
import plotly.offline as plyo
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

#import mysql.connector as sql
import pandas as pd
import numpy as np    
import datetime
from datetime import datetime
from datetime import date 


#import mysql.connector
#from sqlalchemy import create_engine

#import time

import pathlib
import os

#get_ipython().run_line_magic('matplotlib', 'inline')

#quandl.ApiConfig.api_key="Fu-PyTRnxe9TXGGYvEcR"



#create connection

#db_connection=sql.connect(user='root',password='kogs@mysql',host='127.0.0.1',port='3306',database='kogs_quant')
#db_cursor=db_connection.cursor()





def histogram_chart (df,y_value, x1_value, x2_value):
    
    data=df[df[y_value]==int(df[y_value].loc[df['date']==df['date'].max()])].dropna()   
    x1 = data[x1_value]
    x2=data[x2_value]
    # Group data together
    hist_data = [x1, x2]
    group_labels = [x1_value, x2_value]

    # Create distplot with custom bin_size
    fig = ff.create_distplot(hist_data, group_labels, bin_size=.02)
    fig.update_layout(
        #title_text=y_value,
        paper_bgcolor="rgba(0,0,0,0)",
        #plot_bgcolor="rgba(0,0,0,0)",
        font={"color": '#333333'},
        showlegend=True,
        autosize=True,
        margin=dict(l=80, r=10, t=10, b=10),
        )
    return fig

def x2_chart(df,primary_y,secondary_y,primary_y_name, secondary_y_name):
    
     # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    for i in primary_y:
        fig.add_trace(
    
        go.Scatter(x=df.date, y=df[i], name=primary_y_name +' '+ str(i)),
        secondary_y=False,
        )
    
    fig.add_trace(
    go.Scatter(x=df.date, y=df[secondary_y], name=secondary_y_name),
        secondary_y= True,
        )

        
   
    # Add figure title
    fig.update_layout(
        #title_text="Double Y Axis Example",
        paper_bgcolor="rgba(0,0,0,0)",
        #plot_bgcolor="rgba(0,0,0,0)",
        #legend={"font": {"color": "darkgray"}, "orientation": "h", "x": 0, "y": 1.1},
        #font={"color": '#7FDBFF'},
        showlegend=True,
        
    )

    # Set x-axis title
    fig.update_xaxes(title_text="date",showgrid=False)

    # Set y-axes titles
    #primary
    fig.update_yaxes(
        title_text=primary_y_name, 
        secondary_y=False,
        showgrid=False),
    #secondary
    fig.update_yaxes(
        title_text=secondary_y_name, 
        secondary_y=True,
        showgrid=True), 
      
    fig.update_xaxes(rangeslider_visible=True, type='date')
    
    fig.update_xaxes(
        rangeselector=dict(
        buttons=list([
        dict(count=1, label="1M", step="month", stepmode="backward"),
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(count=3, label="3Y", step="year", stepmode="backward"),
        dict(step="all")
        ])
        )
    )
    
    return fig

def discrete_background_color_bins(df, n_bins=5, columns='all'):
    import colorlover
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    if columns == 'all':
        if 'id' in df:
            df_numeric_columns = df.select_dtypes('number').drop(['id'], axis=1)
        else:
            df_numeric_columns = df.select_dtypes('number')
    else:
        df_numeric_columns = df[columns]
    df_max = df_numeric_columns.max().max()
    df_min = df_numeric_columns.min().min()
    ranges = [
        ((df_max - df_min) * i) + df_min
        for i in bounds
    ]
    styles = []
    legend = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        backgroundColor = colorlover.scales[str(n_bins)]['seq']['Blues'][i - 1]
        color = 'white' if i > len(bounds) / 2. else 'inherit'

        for column in df_numeric_columns:
            styles.append({
                'if': {
                    'filter_query': (
                        '{{{column}}} >= {min_bound}' +
                        (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
                    ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                    'column_id': column
                },
                'backgroundColor': backgroundColor,
                'color': color
            })
        legend.append(
            html.Div(style={'display': 'inline-block', 'width': '60px'}, children=[
                html.Div(
                    style={
                        'backgroundColor': backgroundColor,
                        'borderLeft': '1px rgb(50, 50, 50) solid',
                        'height': '10px'
                    }
                ),
                html.Small(round(min_bound, 2), style={'paddingLeft': '2px'})
            ])
        )

    return (styles, html.Div(legend, style={'padding': '5px 0 5px 0'}))


    

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#APP_PATH = str(pathlib.Path(__file__).parent.resolve())
APP_PATH=os.path.abspath('')

#APP_PATH=pathlib.Path(__file__).parent.absolute()
#test = pd.read_csv(os.path.join(APP_PATH, os.path.join("data", "finc_chart.csv")))


#Import data files

rm_data=pd.read_csv('data/regime_model.csv', index_col=0, parse_dates=True)
rm_data['date'] = pd.to_datetime(rm_data['date'])

m_vol_breadth=pd.read_csv('data/m_vol_breadth_chart.csv', index_col=0, parse_dates=True)
m_vol_breadth=m_vol_breadth.reset_index()


fin_cond = pd.read_csv('data/finc_chart.csv', index_col=0, parse_dates=True)
fin_cond=fin_cond.reset_index()

sml_cap_model= pd.read_csv('data/small_cap_model.csv', index_col=0, parse_dates=True)
sml_cap_model=sml_cap_model.reset_index()

rs_model=pd.read_csv('data/rs_model.csv', index_col=0, parse_dates=True)
rs_model=rs_model.reset_index()
rs_model['date'] = pd.to_datetime(rs_model['date']).dt.date
rs_model=rs_model.tail(50)

rs_model_sprd=pd.read_csv('data/rs_model_sprd.csv', index_col=0, parse_dates=True)
rs_model_sprd=rs_model_sprd.reset_index()
rs_model_sprd['date'] = pd.to_datetime(rs_model_sprd['date']).dt.date
rs_model_sprd=rs_model_sprd.tail(50)

hy_sprd=pd.read_csv('data/hy_sprd.csv', index_col=0, parse_dates=True)
hy_sprd=hy_sprd.reset_index()

valuation_sprd=pd.read_csv('data/valuation_sprd.csv', index_col=0, parse_dates=True)
valuation_sprd=valuation_sprd.reset_index()

ind_prod=pd.read_csv('data/ind_prod.csv', index_col=0, parse_dates=True)
ind_prod=ind_prod.reset_index()

curncy=pd.read_csv('data/curncy.csv', index_col=0, parse_dates=True)
curncy=curncy.reset_index()

rel_corr=pd.read_csv('data/rel_corr.csv', index_col=0, parse_dates=True)
rel_corr=rel_corr.reset_index()

crosses=pd.read_csv('data/crosses.csv', index_col=0, parse_dates=True)
crosses=crosses.reset_index()



df=rm_data.set_index('date').dropna()
df['rm_score']=100*(df['RM_Score']/10)
df['technical_score']=100*(df['market_cross']+df['mb_signal']+df['correlation_signal'])/3
df['volatility_score']=100*(df['market_vol_cross']+df['vol_breadth_signal'])/2
df['economic_growth_score']=100*(df['indpro_signal']+df['audchf_signal'])/2
df['financial_conditions_score']=100*(df['fc_signal']+df['hy_signal'])/2
df['sentiment_score']=100*(df['valuation_spread_signal'])/1
df=df.resample('W').pad()

df['1M']=df['Adj Close'].shift(-4)/df['Adj Close']-1
df['3M']=df['Adj Close'].shift(-12)/df['Adj Close']-1

df=df.reset_index()





colors = {
    'background': '#F1F8FB',
    'text': '#333333'
}

container={
                                'vertical-align': 'top',
                                #'display': 'inline',
                                'display': 'inline-block',
                                'font-family': 'sans-serif',
                                #'color': '#7FDBFF',
                                'padding': '10px',
                                'margin': '10px',
                                #'width': '260px',
                                'width': '45%',
                                'border-radius': '10px',
                                'border': 'solid 1px #333333',
                                'text-align':'center',
                                #'background-color':'#333333',
                                'align':'right',
                                #'display':'flex',
                                'min-width': '600px',
                                'position':'relative',
                                'box-sizing': 'border-box',
                                #'height':'1000px'
                            }
container_1={
                                'vertical-align': 'top',
                                #'display': 'inline',
                                #'display': 'inline-block',
                                #'font-family': 'sans-serif',
                                'font_color': '#7FDBFF',
                                'padding': '10px',
                                'margin': '10px',
                                #'width': '260px',
                                'width': '98%',
                                'border-radius': '10px',
                                'border': 'solid 2px #dddddd',
                                'text-align':'center',
                                #'background-color':'#333333',
                                'align':'right',
                                'display':'flex',
                                #'min-width': '600px',
                                'position':'relative',
                                'box-sizing': 'border-box',
                                'columns':'2'
                            }

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid white',
    'borderTop': '1px solid white',
    'padding': '6px',
    'backgroundColor': '#2B67B6',
    'height': '50px',
    'color': 'white',
}

tab_selected_style = {
    'borderTop': '6px solid #76FA01',
    'borderBottom': '1px solid #06D5F7',
    'backgroundColor': '#2B67B6',
    'color': '#76FA01',
    'padding': '6px',
    'height': '50px',
    'fontWeight': 'bold',
}

(styles, legend) = discrete_background_color_bins(rs_model)
(styles_1, legend_1) = discrete_background_color_bins(rs_model_sprd)


app.layout= html.Div(style={'backgroundColor': colors['background'],'color': colors['text']},children=[


            
    html.Div(
        id="banner",
        style={
        'height':'150px',
        'color':'white', 
        'padding': '15px',
        'border': 'solid 1px #dddddd',
        'background-image':'url(assets/Background-banner-1.jpg)',
       
        },
        children=[
            html.Div(
                id="banner-text",
                children=[
                    html.H1("QRIO"),
                    #html.H6("Equity Regime Model Dashboard"),
                ],
            ),

        ]),    
    
            

    html.Div(children=[
        
        dcc.Tabs(id='Tabs', children=[
            
            dcc.Tab( label='US Regime Model', style=tab_style, selected_style=tab_selected_style,children=[
                

                
                html.Div(style=container,
                    
                    children=[
                                                
                      html.Div(style=container_1, children=[ 
                          
                            daq.Gauge(
                                id="Regime Model Score gauge",
                                color={"gradient":True,"ranges":{"green":[60,100],"yellow":[50,60],"red":[0,50]}},
                                label='Regime Model Score',
                                max=100,
                                min=0,
                                size=200,
                                value=int(df['rm_score'].loc[df['date']==df['date'].max()])
                                
                                        ),
                          
                            dcc.Graph(
                                id='rm_distr_chart',
                                figure=histogram_chart(df,'rm_score','1M','3M')
                                       ),
                          
                            
                        ]),
                      
                        html.Div(style=container_1, children=[ 
                            
                        daq.Gauge(
                                id="Technical Score gauge",
                                color={"gradient":True,"ranges":{"green":[60,100],"yellow":[50,60],"red":[0,50]}},
                                label='Technicals Score',
                                max=100,
                                min=0,
                                size=200,
                                value=int(df['technical_score'].loc[df['date']==df['date'].max()])
                            ),
                          
                        
                            dcc.Graph(
                                id='tech_distr_chart',
                                figure=histogram_chart(df,'technical_score','1M','3M')
                                ),
                            ]),
                      
                    
                html.Div(style=container_1, children=[ 
                        daq.Gauge(
                            id="Fin Cond Score gauge",
                            color={"gradient":True,"ranges":{"green":[60,100],"yellow":[50,60],"red":[0,50]}},
                            label='Financial Conditions Score',
                            max=100,
                            min=0,
                            size=200,
                            value=int(df['financial_conditions_score'].loc[df['date']==df['date'].max()]) 
                           
                             ),
                        dcc.Graph(
                                id='finc_distr_chart',
                                figure=histogram_chart(df,'financial_conditions_score','1M','3M')    
                                ),
                            ]),

                html.Div(style=container_1, children=[ 
                        daq.Gauge(
                            id="Economic Growth Score gauge",
                            color={"gradient":True,"ranges":{"green":[60,100],"yellow":[50,60],"red":[0,50]}},
                            label='Economic Growth Score',
                            max=100,
                            min=0,
                            size=200,
                            value=int(df['economic_growth_score'].loc[df['date']==df['date'].max()])
                           
                             ),
                        dcc.Graph(
                                id='econ_distr_chart',
                                figure=histogram_chart(df,'economic_growth_score','1M','3M')

                                ),
                        ]),
                        
                        
                 html.Div(style=container_1, children=[ 
                        daq.Gauge(
                            id="Volatility Score gauge",
                            color={"gradient":True,"ranges":{"green":[60,100],"yellow":[50,60],"red":[0,50]}},
                            label='Volatility Score',
                            max=100,
                            min=0,
                            size=200,
                            value=int(df['volatility_score'].loc[df['date']==df['date'].max()])
                           
                             ),
                     
                        dcc.Graph(
                                id='vol_distr_chart',
                                figure=histogram_chart(df,'volatility_score','1M','3M')
       
                                ),
                        ]),       
                        
                  
                html.Div(style=container_1, children=[ 
                        daq.Gauge(
                            id="Sentiment Score gauge",
                            color={"gradient":True,"ranges":{"green":[60,100],"yellow":[50,60],"red":[0,50]}},
                            label='Sentiment Score',
                            max=100,
                            min=0,
                            size=200,
                            value=int(df['sentiment_score'].loc[df['date']==df['date'].max()])
                           
                             ),
                        dcc.Graph(
                                id='sent_distr_chart',
                                figure=histogram_chart(df,'sentiment_score','1M','3M')
                                    
                                ),
                        ]),
                        

    ]),
                
    html.Div(style=container, children=[
        
        dcc.Tabs(id='Tabs_1', children=[
            
            dcc.Tab( label='Regime Model', style=tab_style, selected_style=tab_selected_style, children=[
                        
                            
                dcc.Graph(
                    #id="2axes",
                    figure=x2_chart(df,['Adj Close'],'rm_score','S&P 500','US Regime Model Score')
                        ),                       
                            
                            
                dcc.Graph(
                    #id="2axes_2",
                    figure=x2_chart(df,['Adj Close'],'technical_score','S&P 500','Technical Composite Score')
                        ),
                                                               
                dcc.Graph(
                    #id="2axes_3",
                    figure=x2_chart(df,['Adj Close'],'financial_conditions_score','S&P 500','Financial Conditions Composite Score')
                        ),
                          

                dcc.Graph(
                    #id="2axes_4",
                    figure=x2_chart(df,['Adj Close'],'economic_growth_score','S&P 500','Economic Growth Composite Score')
                        ),
                        
                dcc.Graph(
                    #id="2axes_5",
                    figure=x2_chart(df,['Adj Close'],'volatility_score','S&P 500','Volatility Composite Score')
                        ),
                        
                dcc.Graph(
                    #id="2axes_6",
                    figure=x2_chart(df,['Adj Close'],'sentiment_score','S&P 500','Sentiment Composite Score')
                        ),
            ]),
            
            dcc.Tab(label='Technicals',style=tab_style, selected_style=tab_selected_style, children=[
                dcc.Graph(
                    #id="2axes",
                    figure=x2_chart(df,['Adj Close'],'technical_score','S&P 500','Technical Composite Score')
                        ),
                dcc.Graph(
                    #id="2axes_2",
                    figure=x2_chart(m_vol_breadth,['market_breadth'],'mb_signal','Market Breadth','Signal')
                        ),
                dcc.Graph(
                    #id="2axes_3",
                    figure=x2_chart(rel_corr,['Roll_Corr','200dma'],'correlation_signal','Cross Correlations','Corr Signal')
                        ),
                dcc.Graph(
                    #id="2axes_3",
                    figure=x2_chart(crosses,['21d','200d'],'market_cross','Moving Averages','Market Cross Signal')
                        ),
                    
                ]), 
            
            dcc.Tab(label='Financial Conditions', style=tab_style, selected_style=tab_selected_style, children=[
                    
                   dcc.Graph(
                    figure=x2_chart(df,['Adj Close'],'financial_conditions_score','S&P 500','Financial Conditions Composite Score')
                        ),
                
                
                dcc.Graph(
                    
                    figure=x2_chart(fin_cond,['NFCI','21dma','200dma'],'fc_signal','NFCI','Signal')
                        ),
                
                dcc.Graph(
                    figure=x2_chart(hy_sprd,['BAMLH0A0HYM2','21dma','200dma'],'hy_signal','HY Spread','HY Spread Signal')
                        ),                                        
                                                                                
                ]), 
                
            dcc.Tab(label='Economic Growth',style=tab_style, selected_style=tab_selected_style, children=[
                
                dcc.Graph(
                        figure=x2_chart(df,['Adj Close'],'economic_growth_score','S&P 500','Economic Growth Composite Score')
                        ),
                dcc.Graph(
                        figure=x2_chart(ind_prod,['INDPRO','200dma'],'indpro_signal','Industrial Production','Signal')
                        ),
                dcc.Graph(
                        figure=x2_chart(curncy,['Adj Close','21dma','200dma'],'audchf_signal','AUDCHF','Currency Signal')
                        ),
            ]), 
            
            dcc.Tab(label='Volatility',style=tab_style, selected_style=tab_selected_style, children=[
                
                dcc.Graph(
                    
                    figure=x2_chart(df,['Adj Close'],'volatility_score','S&P 500','Volatility Composite Score')
                        ),
                
                dcc.Graph(
                    
                    figure=x2_chart(m_vol_breadth,['vol_breadth'],'vol_breadth_signal','Volatility Breadth','Signal')
                        ),
                
                dcc.Graph(
                    
                    figure=x2_chart(crosses,['21d_vol','200d_vol'],'market_vol_cross','Volatility','Signal')
                        ),
                
            ]), 
            
            dcc.Tab(label='Sentiment',style=tab_style, selected_style=tab_selected_style, children=[
                
                dcc.Graph(
                   
                    figure=x2_chart(df,['Adj Close'],'sentiment_score','S&P 500','Sentiment Composite Score')
                        ),
                
                dcc.Graph(
                  
                    figure=x2_chart(valuation_sprd,['b2p_val_spread','b2p_z'],'valuation_spread_signal','B2P Spread','Signal')
                        ),
                
            ]),
            
        ]),
             
    ]),
            
]),
            dcc.Tab( label='US Small Cap Model', style=tab_style, selected_style=tab_selected_style,children=[
                
                html.Div(style=container,children=[
                    
                    dcc.Graph(
                    figure=x2_chart(sml_cap_model,['comdty/tb_yield','125dma'],'signal','Commodity/TB Yld','Signal')
                            ),
                ]),
            ]),
            
            dcc.Tab( label='EAFE Regime Model',style=tab_style, selected_style=tab_selected_style, children=[]),
            dcc.Tab( label='EM Regime Model', style=tab_style, selected_style=tab_selected_style,children=[]),
            
            dcc.Tab( label='Relative Strength Models', style=tab_style, selected_style=tab_selected_style,children=[
                
                html.Div(style=container,children=[
                    html.Div(legend,style={'float':'right'}),
                        table.DataTable(
                        id='table',
                        columns=[{"name": i, "id": i} for i in rs_model.columns],
                        data=rs_model.to_dict('records'),
                        sort_action='native',
                        page_size=20,
                        fixed_rows={'headers': True},

                        style_table={'height': '600px', 'overflowY': 'auto'},

                        style_cell={'width':'30px',
                                    'textAlign': 'center',
                                    'backgroundColor': 'gray',
                                    'color': 'white'
                                   },
                        style_data_conditional=styles,

                        #style_header={'backgroundColor': '#333333','fontWeight': 'bold','border': '1px solid pink'},
                        style_header={'backgroundColor': '#333333'},
                        style_data={'fontWeight': 'bold'}
                        ),       
        
                        ]),
                
                html.Div(style=container,children=[
                    html.Div(legend_1,style={'float':'right'}),
                        table.DataTable(
                        id='table_1',
                        columns=[{"name": i, "id": i} for i in rs_model_sprd.columns],
                        data=rs_model_sprd.to_dict('records'),
                        sort_action='native',
                        page_size=20,
                        fixed_rows={'headers': True},

                        style_table={'height': '600px', 'overflowY': 'auto'},

                        style_cell={'width':'30px',
                                    'textAlign': 'center',
                                    'backgroundColor': 'gray',
                                    'color': 'white'
                                   },
                        style_data_conditional=styles_1,

                        #style_header={'backgroundColor': '#333333','fontWeight': 'bold','border': '1px solid pink'},
                        style_header={'backgroundColor': '#333333'},
                        style_data={'fontWeight': 'bold'}
                        ),       
        
                        ]),
                
                
            ]),
       

            ]),
        ]),
    ])

                    







 





    
  
# Running the server
if __name__ == "__main__":
    
    app.run_server(port=8060)
    #app.run_server(mode='jupyterlab')


# In[ ]:




