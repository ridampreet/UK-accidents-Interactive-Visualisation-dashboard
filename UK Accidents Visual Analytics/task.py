# Import all the necessary dependables
import sys
from pathlib import Path
from random import random

import numpy
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans


from pathlib import Path
from typing import Tuple, List

import dash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Output, Input, State
from matplotlib.widgets import Button, Slider
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc


def task():
    """
    For the last assignment, there is only one task, which will use your knowledge from all previous assignments.
    If your methods of a1, a2 and a3 are well implemented, a4 will be fairly simple, so reuse the methods if possible for your own
    benefit! If you prefer, you can reimplement any logic you with in the assignment4 folder.

    For this task, feel free to create new files, modules or methods in this file as you see fit. Our test will be done by calling this
    task() method, and we expect to receive the dash app back (similar to a3) and we will run it. No other method will be called by us, so
    make sure your code is running as expected. We will basically run this code: `task().run_server(debug=True)`

    For this task, you will build a dash app where we can perform a simple form of interactivity on it. We will use the accidents.csv
    dataset. This accidents.csv dataset has information about traffic accidents in the UK, and is available to you now.
    You will show the most accident prone areas in the UK on a map, so the regular value shown in the map should let us know the number of accidents
    that occurred in a specific area of the map (which should include the accident's severity as well as a weight factor). That said, the purpose of
    this map is to be used by the police to identify how they can do things better.

    **This is raw data, so preprocess the data as per requirement. Drop columns that you feel are unnecessary for the analysis or clustering.
    Don't forget to add comments about why you did what you did**


    ##############################################################
    # Your task will be to Implement all the below functionalities
    ##############################################################

    1. (40pts) Implement a map with the GPS points for the accidents per month. Have a slider(#slider1) that can be used to filter accident data for the month I need.
        You are free to choose a map style, but I suggest using a scatter plot map.

    2. (10pts) Implement a dropdown to select few of the numerical columns in the dataset that can be used meaningfully to represent the size of the GPS points.
        By default the size of the GPS point on map should be based on the value of "accident_severity" column.

    3. (30pts) You have to Cluster the points now. Be sure to have a text somewhere on the webpage that says what clustering algorithm you are using (e.g. KMeans, dbscan, etc).
        For clustering, you should run a clustering method over the dataset (it should run fairly fast for quick iteration, so make sure to use a simple clustering procedure)
        **COLOR** the GPS points based on the clusetring label returned from the algorithm.

    4. (10pts) Have a button(#run_clustering) to run or rerun the clustering algorithm over the filtered dataset (filtered using #slider1 to select months).

    5. (10pts) At least one parameter of the clustering algorithm should be made available for us to tinker with as a button/slider/dropdown.
        When I change it and click #run_clustering button, your code should rerun the clustering algorithm.
        example: change the number of clusters in kmeans or eps value in dbscan.

        Please note: distance values should be in meters, for example dbscan uses eps as a parameter. This value should be read in mts from users and converted appropriately to be used in clustering,
        so input_eps=100 should mean algorithm uses 100mts circle for finding core and non-core points.

    The total points is 100pts
    """

    # Map used,citation:https: // plotly.com / python / mapbox - layers /
    df3 = pd.read_csv('accidents.csv')
    req_cols_for_clustering = ["latitude", "longitude", "police_force", "accident_severity", "number_of_vehicles",
                               "number_of_casualties", "speed_limit", "light_conditions", "weather_conditions",
                               "road_surface_conditions","urban_or_rural_area"]#columns that have been chosen to perform the clustering

    options_for_dropdown=["police_force", "accident_severity", "number_of_vehicles",
                               "number_of_casualties","urban_or_rural_area",
                               ]#options that will be displayed to set the size of the point on the map, have choosen these columns as they display the most meaningful visualisation, the size represents the severity, casualities, and much more




    # df3 = df3.drop(columns=["longitude","latitude","date","local_authority_district","junction_detail","speed_limit","junction_control","pedestrian_crossing_physical_facilities","pedestrian_crossing_human_control","second_road_number","road_surface_conditions","special_conditions_at_site","carriageway_hazards","trunk_road_flag"], axis=1)
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1("Finding clusters using KMEANS in the UK Accident Dataset",style={"margin-left":"21%"}),
        html.Div(
            dcc.Graph(id="map", style={"margin-left": "20%", "width": "100%"})
        ),
        html.Div([
            dcc.Slider(#slider for the months in the dataframe
                id='slider',
                min=1,
                max=12,
                step=1,
                value=10,
                marks={
                    1: "January",
                    2: "February",
                    3: "March",
                    4: "April",
                    5: "May",
                    6: "June",
                    7: "July",
                    8: "August",
                    9: "September",
                    10: "October",
                    11: "November",
                    12: "December"
                },

            ),
            html.H4(id="test"),

        ], style={'width': '62%', 'display': 'inline-block', "marginLeft": "20%"})
        ,
        html.Hr(style={"margin-left": "20%", "margin-right": "20%"}),
        dcc.Dropdown(
            id='dropdown',
            options=[{"label": x, "value": x}
                     for x in options_for_dropdown],
            value='accident_severity', style={"margin-left": "12%", "width": "77%"}
        ),
        html.Hr(style={"margin-left": "20%", "margin-right": "20%"}),
        html.Div([
            dcc.Slider(#slider for selecting the clusters
                id='clustering',
                min=1,
                max=12,
                step=1,
                value=2,
                marks={
                    1: "1",
                    2: "2",
                    3: "3",
                    4: "4",
                    5: "5",
                    6: "6",
                    7: "7",
                    8: "8",
                    9: "9",
                    10: "10",
                    11: "11",
                    12: "12"
                }
            ),
            html.Div(id='clustering-1')
        ], style={'width': '62%', 'display': 'inline-block', "marginLeft": "20%"}),
        html.Button('Get Results on Map', id='show-map',
                    style={"margin-left": "40%", "height": "100px", "width": "200px", "width": "20%"})
    ])

    @app.callback(#callback will take input as the event of button click and then will return the updated figure and the number on the slider
        Output("map", "figure")
,       [Input('show-map', "n_clicks")],
        [dash.dependencies.State('slider', 'value'), dash.dependencies.State("dropdown", "value"),
         dash.dependencies.State("clustering", "value")])
    def update_output(butn, value, drop_value, clusters_n):
        df = pd.read_csv("accidents.csv")#reading the dataset
        dataframe_2 = df.copy()#making a copy of the dataset

        dataframe_2["month"] = pd.to_numeric(pd.DatetimeIndex(dataframe_2["date"]).month)#fetching the month from the dataset so that points on the map can be displayed according to the month selected from the slider


        dataframe_2 = dataframe_2[dataframe_2["month"] == value]#value contains the month that has been selected on the slider, now we use the month selected on the slider to get the data points for that specific month

        model = KMeans(n_clusters=clusters_n)#use kmeans to perform the clustering
        for i in dataframe_2.columns:

            if i not in req_cols_for_clustering and i!="date":#removing unwanted columns from the dataframe


                    dataframe_2 = dataframe_2.drop(columns=i, axis=1)


        dataframe_2 = dataframe_2.drop(columns="date", axis=1)


        dataframe_2=dataframe_2.dropna()#removing NaN

        clusters = model.fit(dataframe_2)
        labels=model.labels_#labels will now be used to provide colors to the clusters

        if drop_value == "accident_severity":#for default value
            fig = px.scatter_mapbox(dataframe_2, lat="latitude", lon="longitude", zoom=4.8, height=1000, width=1300,
                                    size="accident_severity", color=labels)
        else:
            fig = px.scatter_mapbox(dataframe_2, lat="latitude", lon="longitude", zoom=4.8, height=1000, width=1300,
                                    size=str(drop_value), color=labels)

        fig.update_layout(mapbox_style="open-street-map",margin={"r": 10, "t": 10, "l": 10, "b": 10})


        return  fig

    return app








if __name__ == "__main__":
    """
    In case you don't know, this if statement lets us only execute the following lines
    if and only if this file is the one being executed as the main script. Use this
    in the future to test your scripts before integrating them with other scripts.
    """

    task().run_server(debug=True)