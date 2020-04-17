import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objs as go
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy import stats
import math
from sklearn import metrics
from scipy.stats import shapiro, skew, kurtosis

from dash.dependencies import Input, Output, State

def generate_table(df, page_size=10):
    return dash_table.DataTable(
        id = 'dataTable',
        columns = [{
            "name": i, 
            "id": i
        } for i in df.columns],
        data = df.to_dict('records'),
        page_action = "native",
        page_current = 0,
        page_size = page_size
    )

steam = pd.read_csv('Steam_Store_3.csv')
df = pd.DataFrame(steam)
df2 = df.drop(columns= ['release_date', 'developer', 'achievements', 'categories', 'genres', 'steamspy_tags', 'detailed_description', 
                        'positive_ratings', 'negative_ratings', 'mid_range_owners', 'platform_windows', 'platform_mac', 'platform_linux', 
                        'desc_total_word', 'pc_minimum_requirement', 'pc_recommended_requirement', 'mac_minimum_requirement', 
                        'mac_recommended_requirement', 'linux_minimum_requirement', 'linux_recommended_requirement','low_lim_owners', 
                        'up_lim_owners', 'total_of_tag']).copy()

rel_year = df2['release_year'].unique().tolist()
rel_year.sort(reverse=True)
def rel_year_opt(rel_year):
    rel_year_list = []
    for i in rel_year:
        rel_year_dict = {}
        rel_year_dict['label'] = i
        rel_year_dict['value'] = i
        rel_year_list.append(rel_year_dict)
    rel_year_dict = {}
    rel_year_dict['label'] = 'None'
    rel_year_dict['value'] = 'None'
    rel_year_list.append(rel_year_dict)    
    return rel_year_list
year_opt = rel_year_opt(rel_year)

publisher_list = df2['publisher'].value_counts().index.tolist()
def publisher_list_opt(publisher_list):
    publish_list = []
    for i in publisher_list:
        publish_dict = {}
        publish_dict['label'] = i
        publish_dict['value'] = i
        publish_list.append(publish_dict)
    publish_dict = {}
    publish_dict['label'] = 'None'
    publish_dict['value'] = 'None'
    publish_list.append(publish_dict)
    return publish_list
publisher_opt = publisher_list_opt(publisher_list)

top20_publisher = []
for i in range(20):
    item = df2['publisher'].value_counts().index[i]
    top20_publisher.append(item)

new_steam = df.drop(columns=['appid', 'name', 'release_date', 'developer', 'categories', 'genres', 'steamspy_tags', 'detailed_description', 
                            'release_season','positive_ratings', 'negative_ratings', 'release_year', 'mid_range_owners', 'platform_windows', 
                            'mac_minimum_requirement', 'mac_recommended_requirement', 'linux_minimum_requirement', 
                            'linux_recommended_requirement', 'english', 'total_of_tag', 'pc_minimum_requirement', 'platform_linux',
                            'pc_recommended_requirement', 'price', 'desc_total_word','platform_mac','achievements']).copy()

def publisher_filtered(top20_publisher):
    filter_publisher = []

    for i in top20_publisher:
        data = new_steam[new_steam['publisher'] == i]
        target_low = data['low_lim_owners']
        target_up = data['up_lim_owners']
    
        x_train_low5, x_test_low5, y_train_low5, y_test_low5 = train_test_split(data.drop(columns=['publisher', 'low_lim_owners', 'up_lim_owners']), target_low, test_size = 0.3, random_state = 101)
        x_train_up5, x_test_up5, y_train_up5, y_test_up5 = train_test_split(data.drop(columns=['publisher', 'low_lim_owners', 'up_lim_owners']), target_up, test_size = 0.3, random_state = 101)
    
        lm_steam_low5 = LinearRegression()
        lm_steam_up5 = LinearRegression()
        lm_steam_low5.fit(x_train_low5, y_train_low5)
        lm_steam_up5.fit(x_train_up5, y_train_up5)
    
        predictions_low5 = lm_steam_low5.predict(x_train_low5)
        predictions_up5 = lm_steam_up5.predict(x_train_up5)
        
        r2_low5 = metrics.r2_score(y_train_low5, predictions_low5)
        r2_up5 = metrics.r2_score(y_train_up5, predictions_up5)
    
        if (r2_low5 > 0.7) and (r2_low5 < 1) and (r2_up5 < 1) and (r2_up5 < 1):
            filter_publisher.append(i)
    return filter_publisher
filter_publisher = publisher_filtered(top20_publisher)

def filter_publisher_opt(filter_publisher):
    filter_publish_list = []
    for i in filter_publisher:
        filter_publish_dict = {}
        filter_publish_dict['label'] = i
        filter_publish_dict['value'] = i
        filter_publish_list.append(filter_publish_dict)
    filter_publish_dict = {}
    filter_publish_dict['label'] = 'None'
    filter_publish_dict['value'] = 'None'
    filter_publish_list.append(filter_publish_dict)
    return filter_publish_list
filter_publisher_option = filter_publisher_opt(filter_publisher)

def calculate_model(publisher):
    coef_low = {}
    coef_up = {}
    
    data_x2 = new_steam[new_steam['publisher'] == publisher]
    target_low = data_x2['low_lim_owners']
    target_up = data_x2['up_lim_owners']
    
    x_train_low6, x_test_low6, y_train_low6, y_test_low6 = train_test_split(data_x2.drop(columns=['publisher', 'low_lim_owners', 'up_lim_owners']), target_low, test_size = 0.3, random_state = 101)
    x_train_up6, x_test_up6, y_train_up6, y_test_up6 = train_test_split(data_x2.drop(columns=['publisher', 'low_lim_owners', 'up_lim_owners']), target_up, test_size = 0.3, random_state = 101)
    
    lm_steam_low6 = LinearRegression()
    lm_steam_up6 = LinearRegression()
    lm_steam_low6.fit(x_train_low6, y_train_low6)
    lm_steam_up6.fit(x_train_up6, y_train_up6)
    
    coef_low['intercept1'] = lm_steam_low6.intercept_
    coef_up['intercept2'] = lm_steam_up6.intercept_
    
    for i in range(len(x_train_low6.columns)):
        coef_low[x_train_low6.columns[i]] = lm_steam_low6.coef_[i]
    for i in range(len(x_train_up6.columns)):
         coef_up[x_train_up6.columns[i]] = lm_steam_up6.coef_[i]
            
    return [coef_low, coef_up]

def est_param(mean_x, std_x, len_x, ci):
    if len_x < 30:
        critical_value = stats.t.ppf(ci, len_x-1)
    else:
        critical_value = stats.norm.ppf(ci)

    margin_of_error = critical_value*(std_x/math.sqrt(len_x))
    lower = mean_x-margin_of_error
    upper = mean_x+margin_of_error
    
    return [int(round(lower)), int(round(upper))]

def bootstrap(n_size, n_boot, ci, data):
    mean_list = list()
    
    for i in range(n_boot):
        resamp_data = data.sample(n = n_size, replace=True)
        mean_resamp = resamp_data.mean()
        mean_list.append(mean_resamp)
    
    a = (1.0-ci)/2.0
    
    #confidence interval
    p1 = (a)*100
    lower = np.percentile(mean_list,p1)
    p2 = (ci+(a))*100
    upper = np.percentile(mean_list,p2)
    
    return [int(round(lower)), int(round(upper))]

def get_model(publisher):
    coef_low = calculate_model(publisher)[0]
    coef_up = calculate_model(publisher)[1]
    y_formula_low = ''
    y_formula_up = ''
    
    for i,j in zip(coef_low.keys(), coef_low.values()):
        if (i == 'intercept1'):
            y_formula_low = 'Y_low = {:.2f}'.format(j)
        else:
            if str(j)[0] == '-':
                y_formula_low += ' - {:.2f}*X_{}'.format(float(str(j)[1:]), i)
            else:
                y_formula_low += ' + {:.2f}*X_{}'.format(j,i)
    
    for i,j in zip(coef_up.keys(), coef_up.values()):
        if (i == 'intercept2'):
            y_formula_up = 'Y_up = {:.2f}'.format(j)
        else:
            if str(j)[0] == '-':
                y_formula_up += ' - {:.2f}*X_{}'.format(float(str(j)[1:]), i)
            else:
                y_formula_up += ' + {:.2f}*X_{}'.format(j,i)

    return [y_formula_low, y_formula_up]

def owners_prediction(publisher, x1_low, x2_low, x3_low, x4_low, x5_low, x1_up, x2_up, x3_up, x4_up, x5_up):
    coef_low = calculate_model(publisher)[0]
    coef_up = calculate_model(publisher)[1]
    
    coef_x_var_low = coef_low['intercept1'] + coef_low['required_age']*x1_low + coef_low['average_playtime']*x2_low + coef_low['total_of_category']*x3_low + coef_low['total_of_genre']*x4_low + coef_low['ratings_gap']*x5_low
    coef_x_var_up = coef_up['intercept2'] + coef_up['required_age']*x1_up + coef_up['average_playtime']*x2_up + coef_up['total_of_category']*x3_up + coef_up['total_of_genre']*x4_up + coef_up['ratings_gap']*x5_up
    
    return [coef_x_var_low,coef_x_var_up]

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1("Steam Dash"),

    html.Div(children="""
        Steam Dash: A simple web application for 'Steam' data and simple dashboard to apply machine learning using Multiple Linear Regression.
    """),

    html.Br(),
    dcc.Tabs(children = [
        
        dcc.Tab(value = 'Tab1', label = 'Steam Data Frame', children = [
            html.Div([
                html.Div([
                    html.P('Released Year'),
                    dcc.Dropdown(
                                    value='None',
                                    id='filter-year',
                                    options = year_opt
                    )  
                ], className='col-3'),
                html.Div([
                    html.P('Released Season'),
                    dcc.Dropdown(
                                    value='None',
                                    id='filter-season',
                                    options=[
                                        {'label': 'Spring', 'value': 'Spring'},
                                        {'label': 'Summer', 'value': 'Summer'},
                                        {'label': 'Fall', 'value': 'Fall'},
                                        {'label': 'Winter', 'value': 'Winter'},
                                        {'label': 'None', 'value': 'None'}
                                    ] 
                    )  
                ], className='col-3'),
                html.Div([
                    html.P('Publisher'),
                    dcc.Dropdown(
                                    value='None',
                                    id='filter-publisher',
                                    options=publisher_opt
                    )  
                ], className='col-3'),
                html.Div([
                    html.P('English'),
                    dcc.Dropdown(
                                    value='None',
                                    id='filter-english',
                                    options=[
                                        {'label': 'Yes', 'value': 1},
                                        {'label': 'No', 'value': 0},
                                        {'label': 'None', 'value': 'None'}
                                    ] 
                    )  
                ], className='col-3'),
           ],className='row' ),

            html.Br(),
            html.Div([
                html.P('Max Rows'),
                dcc.Input(
                            id='filter-row',
                            value=10
                )  
            ], className= 'row col-3'),
            
            html.Br(),
            html.Div(children = [
                html.Button('search', id='filter')
            ], className= 'row col-3'),

            html.Br(),
            html.Div(id ='div-table',children=[
                    generate_table(df2)
            ], className = 'col-12')
        ]),

        dcc.Tab(value = 'Tab2', label = 'Scatter Plot Example', children = [
            html.Div([
                html.Div([
                    html.P('Hue'),
                    dcc.Dropdown(
                                    value='None',
                                    id='filter-hue',
                                    options=[
                                        {'label': 'Year', 'value': 'release_year'},
                                        {'label': 'Season', 'value': 'release_season'},
                                        {'label': 'Publisher', 'value': 'publisher'},
                                        {'label': 'English', 'value': 'english'},
                                        {'label': 'None', 'value': 'None'}
                                    ] 
                    )  
                ], className='col-3'),

                html.Div([
                    html.P('X-axis'),
                    dcc.Dropdown(
                                    value='None',
                                    id='filter-xaxis',
                                    options=[
                                        {'label': 'Required Age', 'value': 'required_age'},
                                        {'label': 'Average Playtime', 'value': 'average_playtime'},
                                        {'label': 'Price', 'value': 'price'},
                                        {'label': 'Total of Category', 'value': 'total_of_category'},
                                        {'label': 'Total of Genre', 'value': 'total_of_genre'},
                                        {'label': 'Total of Tag', 'value': 'total_of_tag'},
                                        {'label': 'Ratings Gap', 'value': 'ratings_gap'},
                                        {'label': 'None', 'value': 'None'}
                                    ] 
                    )  
                ], className='col-3'),

                html.Div([
                    html.P('y-axis'),
                    dcc.Dropdown(
                                    value='None',
                                    id='filter-yaxis',
                                    options=[
                                        {'label': 'Required Age', 'value': 'required_age'},
                                        {'label': 'Average Playtime', 'value': 'average_playtime'},
                                        {'label': 'Price', 'value': 'price'},
                                        {'label': 'Total of Category', 'value': 'total_of_category'},
                                        {'label': 'Total of Genre', 'value': 'total_of_genre'},
                                        {'label': 'Total of Tag', 'value': 'total_of_tag'},
                                        {'label': 'Ratings Gap', 'value': 'ratings_gap'},
                                        {'label': 'None', 'value': 'None'}
                                    ] 
                    )  
                ], className='col-3'),

                
            ], className='row'),

            html.Br(),
            html.Div(children = [
                html.Button('Create the Plot', id='create_scatter')
            ], className= 'row col-3'),
            
            html.Br(),
            html.Div(id='show_graph', 
                children=dcc.Graph(id='graphs_scatter')
            )
        ]),

        dcc.Tab(value = 'Tab3', label = 'Linear Regression Model & Owners Prediction', children = [
           html.Div([
                html.P('Publisher'),
                dcc.Dropdown(
                                value='None',
                                id='choosen_publisher4',
                                options=filter_publisher_option
                )  
            ], className='col-4'),
            html.Br(),

            html.Div(children = [
                html.Button('Get Model', id='get_linear_model')
            ], className= 'row col-3'),
            html.Br(),

            html.Div(id='output_model_low'),
            html.Br(),
            html.Div(id='output_model_up'),
            
            html.Br(),
            html.Br(),

            html.Div([
                html.Div([
                    html.P('X1_low : '),
                    dcc.Input(
                                id='x1_low',
                                type='number',
                                value= 0
                    )  
                ], className='col-2'),
                html.Div([
                    html.P('X2_low : '),
                    dcc.Input(
                                id='x2_low',
                                type='number',
                                value= 0
                    )  
                ], className='col-2'),
                html.Div([
                    html.P('X3_low : '),
                    dcc.Input(
                                id='x3_low',
                                type='number',
                                value= 0
                    )  
                ], className='col-2'),
                html.Div([
                    html.P('X4_low : '),
                    dcc.Input(
                                id='x4_low',
                                type='number',
                                value= 0
                    )  
                ], className='col-2'),
                html.Div([
                    html.P('X5_low : '),
                    dcc.Input(
                                id='x5_low',
                                type='number',
                                value= 0
                    )  
                ], className='col-2')
            ], className='row'),
            html.Br(),

            html.Div([
                html.Div([
                    html.P('X1_up : '),
                    dcc.Input(
                                id='x1_up',
                                type='number',
                                value= 0
                    )  
                ], className='col-2'),
                html.Div([
                    html.P('X2_up : '),
                    dcc.Input(
                                id='x2_up',
                                type='number',
                                value= 0
                    )  
                ], className='col-2'),
                html.Div([
                    html.P('X3_up : '),
                    dcc.Input(
                                id='x3_up',
                                type='number',
                                value= 0
                    )  
                ], className='col-2'),
                html.Div([
                    html.P('X4_up : '),
                    dcc.Input(
                                id='x4_up',
                                type='number',
                                value= 0
                    )  
                ], className='col-2'),
                html.Div([
                    html.P('X5_up : '),
                    dcc.Input(
                                id='x5_up',
                                type='number',
                                value= 0
                    )  
                ], className='col-2')
            ], className='row'),
            html.Br(),

            html.Div(children = [
                html.Button('Calculate', id='get_range_owners')
            ], className= 'row col-3'),

            html.Br(),

            html.Div(id='output_prediction')                
        ])
    ],
        ## Tabs Content Style
        content_style = {
            'fontFamiliy': 'Arial',
            'borderBottom': '1px solid #d6d6d6',
            'borderLeft': '1px solid #d6d6d6',
            'borderRight': '1px solid #d6d6d6',
            'padding': '44px'
        }
    )
       
],
    #Div paling luar
    style = {
        'maxWidth': '1200px',
        'margin': '0 auto'
    }
)

@app.callback(
    Output(component_id = 'div-table', component_property = 'children'),
    
    [Input(component_id = 'filter', component_property = 'n_clicks')],
    
    [State(component_id = 'filter-row', component_property = 'value'),
    State(component_id = 'filter-year', component_property = 'value'),
    State(component_id = 'filter-season', component_property = 'value'),
    State(component_id = 'filter-publisher', component_property = 'value'),
    State(component_id = 'filter-english', component_property = 'value')]
)
def input_table(n_clicks, row, year, season, publisher, english):
    steam = pd.read_csv('Steam_Store_3.csv')
    df = pd.DataFrame(steam)
    
    df2 = df.drop(columns= ['release_date', 'developer', 'achievements', 'categories', 'genres', 'steamspy_tags', 'detailed_description', 
                        'positive_ratings', 'negative_ratings', 'mid_range_owners', 'platform_windows', 'platform_mac', 'platform_linux', 
                        'desc_total_word', 'pc_minimum_requirement', 'pc_recommended_requirement', 'mac_minimum_requirement', 
                        'mac_recommended_requirement', 'linux_minimum_requirement', 'linux_recommended_requirement', 'low_lim_owners', 
                        'up_lim_owners', 'total_of_tag']).copy()

    if year != 'None':
        df2 = df2[df2['release_year'] == year]
    if season != 'None':
        df2 = df2[df2['release_season'] == season]
    if publisher != 'None':
        df2 = df2[df2['publisher'] == publisher]
    if english != 'None':
        df2 = df2[df2['english'] == english]
    
    children = [generate_table(df2, page_size = int(row))]

    return children

@app.callback(
     Output(component_id = 'show_graph', component_property = 'children'),
    
    [Input(component_id = 'create_scatter', component_property = 'n_clicks')],

    [State(component_id = 'filter-hue', component_property = 'value'),
    State(component_id = 'filter-xaxis', component_property = 'value'),
    State(component_id = 'filter-yaxis', component_property = 'value')]
)
def create_scatter_plot(n_clicks, f_hue, f_xaxis, f_yaxis):
    # scatter_steam = ''
    if ((f_hue == 'None') or (f_hue == '') or (f_hue == 'Null')) or ((f_xaxis == 'None') or (f_xaxis == '') or (f_xaxis == 'Null')) or ((f_yaxis == 'None') or (f_yaxis == '') or (f_yaxis == 'Null')):
        return ''
    else:
        scatter_steam = dcc.Graph(
                            id = 'graph-scatter',
                            figure={
                                'data':[
                                    go.Scatter(
                                        x=df2[df2[f_hue]==i][f_xaxis],
                                        y=df2[df2[f_hue]==i][f_yaxis],
                                        mode='markers',
                                        name= 'Season {}'.format(i)
                                    ) for i in df2[f_hue].unique()
                                ],
                                'layout':
                                    go.Layout(
                                        xaxis={'title': '{}'.format(f_xaxis)},
                                        yaxis={'title': '{}'.format(f_yaxis)},
                                        title='Steam Dash Scatter Visualization',
                                        hovermode='closest'
                                    )
                            }
                        )
        return scatter_steam

@app.callback(
    Output(component_id = 'output_model_low', component_property = 'children'),
    
    [Input(component_id = 'get_linear_model', component_property = 'n_clicks')],

    [State(component_id = 'choosen_publisher4', component_property = 'value')]
)
def get_model_regression_low(n_clicks, choosen_publisher4):
    if (choosen_publisher4 == '') or (choosen_publisher4 == 'Null') or (choosen_publisher4 == 'None'):
        return ''
    else:
        linear_model = get_model(choosen_publisher4)
        return '{}'.format(linear_model[0]) 
@app.callback(
    Output(component_id = 'output_model_up', component_property = 'children'),
    
    [Input(component_id = 'get_linear_model', component_property = 'n_clicks')],

    [State(component_id = 'choosen_publisher4', component_property = 'value')]
)
def get_model_regression_up(n_clicks, choosen_publisher4):
    if (choosen_publisher4 == '') or (choosen_publisher4 == 'Null') or (choosen_publisher4 == 'None'):
        return ''
    else:
        linear_model = get_model(choosen_publisher4)
        return '{}'.format(linear_model[1])

@app.callback(
    Output(component_id = 'output_prediction', component_property = 'children'),
    
    [Input(component_id = 'get_range_owners', component_property = 'n_clicks')],

    [State(component_id = 'choosen_publisher4', component_property = 'value'),
    State(component_id = 'x1_low', component_property = 'value'),
    State(component_id = 'x2_low', component_property = 'value'),
    State(component_id = 'x3_low', component_property = 'value'),
    State(component_id = 'x4_low', component_property = 'value'),
    State(component_id = 'x5_low', component_property = 'value'),
    State(component_id = 'x1_up', component_property = 'value'),
    State(component_id = 'x2_up', component_property = 'value'),
    State(component_id = 'x3_up', component_property = 'value'),
    State(component_id = 'x4_up', component_property = 'value'),
    State(component_id = 'x5_up', component_property = 'value')]
)
def get_prediction(n_clicks, choosen_publisher4, x1_low, x2_low, x3_low, x4_low, x5_low, x1_up, x2_up, x3_up, x4_up, x5_up):
    if (choosen_publisher4 == '') or (choosen_publisher4 == 'Null') or (choosen_publisher4 == 'None'):
        return ''
    else:
        predictions_res = owners_prediction(choosen_publisher4, x1_low, x2_low, x3_low, x4_low, x5_low, x1_up, x2_up, x3_up, x4_up, x5_up)

        return 'So the owners prediction for "{}" publisher is {}-{}'.format(choosen_publisher4, int(predictions_res[0]), int(predictions_res[1]))

if __name__ == '__main__':
    app.run_server(debug=True)