import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import dash_daq as daq

data = pd.read_csv('C:/Users/perso/Desktop/3 year/Data project/Zomato Chennai Listing 2020.csv')
x = data.iloc[:,1:].values

#replacing all the invalid or not filled values in the dataset with NaN
data.replace(to_replace = ['None','Invalid','Does not offer Delivery','Does not offer Dining','Not enough Delivery Reviews',
                           'Not enough Dining Reviews'], value =np.nan,inplace=True)

#converting the names of restaurant to lower case
data['Name of Restaurant'] = data['Name of Restaurant'].apply(lambda x: x.lower())
data['Features'] = data['Features'].apply(lambda x:x.replace('[','').replace(']','').replace("'",'').replace('  ','').split(','))
#converting the top dishes field to string and removing its braces and '
data['Top Dishes'] = data["Top Dishes"].astype(str)
data['Top Dishes'] = data['Top Dishes'].apply(lambda x:x.replace('[','').replace(']','').replace("'",'').replace('  ','').split(','))
data['Cuisine'] = data["Cuisine"].astype(str)
data['Dining Rating Count'] = data['Dining Rating Count'].astype("Float32")
data['Dining Rating'] = data['Dining Rating'].astype("Float32")
data['Delivery Rating Count'] = data['Delivery Rating Count'].astype("Float32")
data['Delivery Rating'] = data['Delivery Rating'].astype("Float32")

def vegs(val):
    if 'Vegetarian Only' in val:
        return 'Yes'
    elif ' Vegetarian Only' in val:
        return 'Yes'
    else:
        return 'No'
data['Vegetarian Status'] = data['Features'].apply(lambda x: vegs(x))
#data = data.iloc[:,1:].values

app = dash.Dash()

app.layout = html.Div([
    html.H1("Analysis of Zomato based Restaurants in Chennai", style = {'text-align':'center', 'color': 'black',
                                                                        'fontSize': 50 , 'font-style': 'italic'}),
    html.Div(
    dcc.Dropdown(
        id='select_region',
        options=[
           {'label': 'Pallavaram', 'value': 'Pallavaram'},
            {'label': 'Alandur', 'value': 'Alandur'},
            {'label': 'Kodambakkam', 'value': 'Kodambakkam'},
            {'label': 'T. Nagar', 'value': 'T. Nagar'},
            {'label': 'Adyar', 'value': 'Adyar'},
            {'label': 'Velachery', 'value': 'Velachery'},
            {'label': 'Porur', 'value': 'Porur'},
            {'label': 'Madipakkam', 'value': 'Madipakkam'},
            {'label': 'Anna Nagar West', 'value': 'Anna Nagar West'},
            {'label': 'Anna Nagar East', 'value': 'Anna Nagar East'}
        ],
        value='Pallavaram' ),
        style = {'margin-top': '5px', 'margin-left' : '690px', 'width': '25%'} 
    ),
    html.Div(id = 'output_container', children = [],style = {'margin-top': '5px', 'margin-left' : '750px', 'width': '20%',
                                                             'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
    html.Br(),
    dcc.Graph(id = 'plot_area', figure = {}),
    html.Br(),
    html.H1("Enter the Price Value", style = {'text-align':'center', 'color': 'blue', 'fontSize': 25}),
    html.Div(
    daq.Slider(
        min=40,
        max=5000,
        id='pricetag',
        value = 1000,
        handleLabel={"showCurrentValue": True,"label": "VALUE"}),
    style={'border': 'solid 1px #A2B1C6', 'border-radius': '5px', 'padding': '5px', 'margin-top': '5px', 
           'margin-left' : '750px', 'width': '20%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
    html.Div(id='slider-output', style = {'margin-top': '5px', 'margin-left' : '750px', 'width': '20%', 
                                          'display': 'flex', 'align-items': 'center', 'justify-content': 'center'},children = []),
    dcc.Graph(id = 'plot_price', figure = {}),
    html.Br(),
    html.H1("Select the Region", style = {'text-align':'center', 'color': 'blue', 'fontSize': 25}),
    html.Div(
    dcc.Dropdown(
        id='select_region_val',
        options=[
            {'label': 'Pallavaram', 'value': 'Pallavaram'},
            {'label': 'Alandur', 'value': 'Alandur'},
            {'label': 'Kodambakkam', 'value': 'Kodambakkam'},
            {'label': 'T. Nagar', 'value': 'T. Nagar'},
            {'label': 'Adyar', 'value': 'Adyar'},
            {'label': 'Velachery', 'value': 'Velachery'},
            {'label': 'Porur', 'value': 'Porur'},
            {'label': 'Madipakkam', 'value': 'Madipakkam'},
            {'label': 'Anna Nagar West', 'value': 'Anna Nagar West'},
            {'label': 'Anna Nagar East', 'value': 'Anna Nagar East'}
        ],
        value='Pallavaram' ),
        style = {'margin-top': '5px', 'margin-left' : '750px', 'width': '25%'} 
    ),
    html.Div(id = 'output', children = [], style = {'margin-top': '5px', 'margin-left' : '750px', 'width': '20%',
                                                    'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
    html.Br(),
    dcc.Graph(id = 'plot_pie', figure = {}),
    html.Br(),
    html.H1("Enter the Dining Rating", style = {'text-align':'center', 'color': 'blue', 'fontSize': 25}),
    html.Div(
    daq.Slider(
        min=1,
        max=4,
        id='dining',
        value = 2,
        handleLabel={"showCurrentValue": True,"label": "VALUE"}),
    style={'border': 'solid 1px #A2B1C6', 'border-radius': '5px', 'padding': '5px', 'margin-top': '5px', 
           'margin-left' : '750px', 'width': '20%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
    html.Div(id='slider-rating', style = {'margin-top': '5px', 'margin-left' : '750px', 'width': '20%', 
                                          'display': 'flex', 'align-items': 'center', 'justify-content': 'center'},children = []),
    dcc.Graph(id = 'plot_rating', figure = {}),
])

@app.callback(
    [dash.dependencies.Output('output_container', 'children'),
     dash.dependencies.Output('plot_area', 'figure')],
    [dash.dependencies.Input('select_region', 'value')])


def update_output(value):
    container = 'Region selected "{}"'.format(value)
    #df = data
    #df = df[df[:,2] == value]
    dd = data
    dd = dd[dd['Location'] == value].sample(frac = 0.2)
    #plotly express
    fig = px.bar(
        data_frame = dd,
        x = 'Name of Restaurant',
        y = 'Price for 2',
        color = 'Vegetarian Status',
        orientation = "v",
        barmode = 'relative' ,
        template = "plotly_white",
        title = "Restaurants in " + value + " wrt price for 2",
        )
    return  container, fig
    
@app.callback(
    [dash.dependencies.Output('slider-output', 'children'),
    dash.dependencies.Output('plot_price', 'figure')],
    [dash.dependencies.Input('pricetag', 'value')]) 
 
def update_out(value):   
    container = 'Price selected "{}"'.format(value)
    dff = data.sample(frac = 0.01)
    dff = dff[dff['Price for 2'] <= value]           
    fig = px.bar(
        data_frame = dff,
        x = 'Name of Restaurant',
        y = 'Price for 2',
        color = 'Vegetarian Status',
        orientation = "v",
        barmode = 'relative' ,
        template = "plotly_white",
        title = "Restaurants within the Price range " + str(value),
        )
    return container, fig

@app.callback(
    [dash.dependencies.Output('output', 'children'),
     dash.dependencies.Output('plot_pie', 'figure')],
    [dash.dependencies.Input('select_region_val', 'value')])


def update_outs(value):
    container = 'Region selected "{}"'.format(value)
    #df = data
    #df = df[df[:,2] == value]
    dd = data
    dd = dd[dd['Location'] == value]
    x = []
    x.append(dd['Vegetarian Status'].value_counts()[0])
    x.append(dd['Vegetarian Status'].value_counts()[1])    
    #plotly express
    fig = px.pie(dd, values= x, labels= 'Vegetarian Status', names = ['No', 'Yes'])
    return  container, fig

@app.callback(
    [dash.dependencies.Output('slider-rating', 'children'),
    dash.dependencies.Output('plot_rating', 'figure')],
    [dash.dependencies.Input('dining', 'value')]) 
 
def update_outp(value):   
    container = 'Rating Chosen "{}"'.format(value)
    dff = data
    if(value >= 2):
        x = 0.01
    else:
        x = 0.1
    dff = dff[dff['Dining Rating'] >= value].sample(frac = x)          
    fig = px.bar(
        data_frame = dff,
        x = 'Name of Restaurant',
        y = 'Dining Rating',
        color = 'Vegetarian Status',
        orientation = "v",
        barmode = 'relative' ,
        template = "plotly_white",
        title = "Restaurants with Dining Rating greater than " + str(value),
        )
    return container, fig

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)