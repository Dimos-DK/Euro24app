## CREATE WEB APP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from mplsoccer import Pitch, VerticalPitch
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import base64
# Load data

# Define the directory containing the CSV files
directory = 'App data/Euro 2024 project/events'

# Find all files in the directory that end with 'events-eurocopa-2024.csv'
file_pattern = os.path.join(directory, '*events-eurocopa-2024.csv')
csv_files = glob.glob(file_pattern)

# Read and concatenate all matching CSV files
events_list = [pd.read_csv(file) for file in csv_files]
events = pd.concat(events_list, ignore_index=True)

# Define the directory containing the CSV files
directory = 'App data/Euro 2024 project/events'

# Find all files in the directory that end with 'events-eurocopa-2024.csv'
file_pattern = os.path.join(directory, '*shots-eurocopa-2024.csv')
csv_files = glob.glob(file_pattern)

# Read and concatenate all matching CSV files
events_list = [pd.read_csv(file) for file in csv_files]
shot_events = pd.concat(events_list, ignore_index=True)

players = pd.read_csv(
    'App data/Euro 2024 project/general_data/all_players-eurocopa-2024.csv')

shots_by_player = pd.read_csv(
    'App data/Euro 2024 project/shots/euro24_shots_by_player.csv')

stats = pd.read_csv(
    'App data/Euro 2024 project/stats/playersStatsSeason-eurocopa-2024.csv')

all_events = pd.read_csv(
    'App data/Euro 2024 project/general_data/euro_events_transformed.csv')

xg_shot_events = pd.read_csv(
    'App data/Euro 2024 project/shots/euro24_xg_shots_splitted.csv')

pd.set_option('display.max_columns', None)

# Initial Transformations

## Create a 'GAME' Column for the xg_shot_events

# Group by 'Match ID' and aggregate the 'Team Name' column
match_teams = xg_shot_events.groupby('Match ID')['Team Name'].unique().reset_index()

# Convert the list of unique team names into the desired format
match_teams['Game'] = match_teams['Team Name'].apply(lambda x: ' vs '.join(sorted(x)))

# Merge the new 'Game' column back into the original DataFrame
xg_shot_events = xg_shot_events.merge(match_teams[['Match ID', 'Game']], on='Match ID', how='left')


# Function to capitalize the first letter of each word in a name
def capitalize_name(name):
    if isinstance(name, str):
        return ' '.join(word.capitalize() for word in name.split())
    return name


# Apply the capitalization function
events['Player From Name'] = events['Player From Name'].apply(capitalize_name)
shot_events['Player From Name'] = shot_events['Player From Name'].apply(capitalize_name)
stats['Known Name'] = stats['Known Name'].apply(capitalize_name)
events['Player To Name'] = events['Player To Name'].apply(capitalize_name)

# Create an English Team name dictionary
team_names = {
    "GEORGIA": "Georgia",
    "ROMANIA": "Romania",
    "AUSTRIA": "Austria",
    "CROAZIA": "Croatia",
    "GERMANIA": "Germany",
    "FRANCIA": "France",
    "POLONIA": "Poland",
    "UCRAINA": "Ukraine",
    "REPUBBLICA CECA": "Czech Republic",
    "PORTOGALLO": "Portugal",
    "TURCHIA": "Turkey",
    "SVIZZERA": "Switzerland",
    "UNGHERIA": "Hungary",
    "SPAGNA": "Spain",
    "ALBANIA": "Albania",
    "SLOVACCHIA": "Slovakia",
    "OLANDA": "Netherlands",
    "ITALIA": "Italy",
    "SERBIA": "Serbia",
    "SLOVENIA": "Slovenia",
    "INGHILTERRA": "England",
    "DANIMARCA": "Denmark",
    "BELGIO": "Belgium",
    "SCOZIA": "Scotland"
}

#Rename the Team names in the English versions
events['Team Name'] = events['Team Name'].replace(team_names)

# Import glossary to rename columns to english
glossary = pd.read_csv(
    'App data/Euro 2024 project/general_data/glossary-euro-2024.csv')

# Create a dictionary for renaming columns
rename_dict = dict(zip(glossary['Code'], glossary['Description']))

# Rename the columns of player_data
stats.rename(columns=rename_dict, inplace=True)

# Unify player stats with shots aggregates
player_stats = pd.merge(stats, shots_by_player, left_on='Player ID', right_on='Player.id', how='left')

# Calculate games played
player_stats['matches'] = (player_stats['Play Time'] / player_stats['Play Time average per game']).fillna(0).astype(int)

# Unify player data with player stats
player_stats = player_stats.merge(players[['Player ID', 'Shirt Number', 'Position', 'Position Detail', 'Citizenship',
                                           'Height', 'Foot', 'Photo', 'Age']], on=['Player ID'], how='left')

# Unify positions in one column
player_stats['Position_sm'] = player_stats['Position Detail'].combine_first(player_stats['Position'])

# Create a position dictionary
positions = {'Center Midfield': 'CM', 'Centre-Back': 'CB', 'Striker': 'ST', 'Right Winger': 'Winger',
             'Left Winger': 'Winger',
             'Midfielder': 'CM', 'Goalkeeper': 'GK', 'Defender': 'CB', 'Defensive Midfield': 'CDM',
             'Left Forward': 'Winger', 'Right-Back': 'RB',
             'Attacking Midfield': 'CAM', 'Centre-Forward': 'ST', 'Left-Back': 'LB'}

# Map positions with dictionary for unification
player_stats['Position_sm'] = player_stats['Position_sm'].map(positions)

# Drop unnecessary columns
player_stats.drop(columns=['First Name', 'Last Name', 'Born Date', 'Season ID', 'Position', 'Position Detail'],
                  inplace=True)

# Rename columns
player_stats = player_stats.rename(
    columns={'Citizenship': 'Team', 'Known Name': 'Player Name', 'Assists': 'Key Passes'})

# Fix Kosovo Citizenship
player_stats['Team'] = player_stats['Team'].replace('Kosovo', 'Albania')

## Create Team aggregated statistics
team_stats = player_stats.groupby(['Team']).agg({
    'Killer Passes': 'sum',
    'Killer Passes in the Box': 'sum',
    'Passes': 'sum',
    'Accurate Passes': 'sum',
    'Passes in the Box': 'sum',
    'Accurate Passes in the Box': 'sum',
    'Key Passes': 'sum',
    'Lost Possessions': 'sum',
    'Ball Recoveries': 'sum',
    'Ball Recoveries in Offensive Half': 'sum',
    'Dribbling': 'sum',
    'Successful Dribbling': 'sum',
    'no-penalty xG': 'sum',
    'xGOT': 'sum',
    'xA': 'sum',
    'open-play xG': 'sum',
    'Shots': 'sum',
    'Shots on Target': 'sum',
    'Shots from Outside the Box': 'sum',
    'Crosses': 'sum',
    'Accurate Crosses': 'sum',
    'Side Balls': 'sum',
    'Offsides': 'sum',
    'Give and Go': 'sum',
    'Ball Recoveries in Zone 3': 'sum',
    'Lost Possessions in Zone 1': 'sum',
    'Goalkeeper Passes': 'sum',
    'Successful Goalkeeper Passes': 'sum',
    'Cut Back': 'sum',
    'Shots on Target Suffered': 'sum',
    'Accurate Cut Back': 'sum',
    'Yellow Cards': 'sum',
    'Red Cards': 'sum',
    'matches': 'max',
    'Goal': 'sum',
    'Big Chance': 'sum'
}
).reset_index()

# Create a new DataFrame for average values
team_stats_avg = team_stats.copy()

# Divide each column (except 'Team' and 'matches') by the 'matches' column and adding the suffix '_avg'
for column in team_stats.columns:
    if column not in ['Team', 'matches']:
        team_stats_avg[column + '_avg'] = team_stats[column] / team_stats['matches']

# Drop the original columns to keep only the average columns and 'Team'
team_stats_avg = team_stats_avg[
    ['Team'] + [col + '_avg' for col in team_stats.columns if col not in ['Team', 'matches']]]

# Create success rates
team_stats_avg['Passing Accuracy'] = (team_stats_avg['Accurate Passes_avg'] / team_stats_avg['Passes_avg']) * 100
team_stats_avg['Crossing Accuracy'] = (team_stats_avg['Accurate Crosses_avg'] / team_stats_avg['Crosses_avg']) * 100
team_stats_avg['Passes in Box Accuracy'] = (team_stats_avg['Accurate Passes in the Box_avg'] / team_stats_avg[
    'Passes in the Box_avg']) * 100
team_stats_avg['GK Passing Accuracy'] = (team_stats_avg['Successful Goalkeeper Passes_avg'] / team_stats_avg[
    'Goalkeeper Passes_avg']) * 100
team_stats_avg['Shots OT pct'] = (team_stats_avg['Shots on Target_avg'] / team_stats_avg['Shots_avg']) * 100
team_stats_avg['Dribbling success'] = (team_stats_avg['Successful Dribbling_avg'] / team_stats_avg[
    'Dribbling_avg']) * 100
team_stats_avg['High Recoveries pct'] = (team_stats_avg['Ball Recoveries in Offensive Half_avg'] / team_stats_avg[
    'Ball Recoveries_avg']) * 100
team_stats_avg['Ball Control Ratio'] = (team_stats_avg['Ball Recoveries_avg'] / team_stats_avg[
    'Lost Possessions_avg']) * 100
team_stats_avg['Goals vs xGoals'] = (team_stats_avg['Goal_avg'] - team_stats_avg['no-penalty xG_avg'])
team_stats_avg['Discipline'] = (team_stats_avg['Yellow Cards_avg'] + team_stats_avg['Red Cards_avg'])

# Round values
team_stats_avg = team_stats_avg.round(1)

team_stats_avg = team_stats_avg.drop(
    columns=['Passes_avg', 'Passes in the Box_avg', 'Dribbling_avg', 'Crosses_avg', 'Goalkeeper Passes_avg',
             'Cut Back_avg', 'Yellow Cards_avg', 'Red Cards_avg'])

### Rank Teams

# Create a new DataFrame for ranked values
team_stats_ranked = team_stats_avg.copy()

# List of metrics that should be ranked in ascending order (inverted ranking)
inverted_rank_metrics = ['Lost Possessions_avg', 'Offsides_avg', 'Lost Possessions in Zone 1_avg', 'Discipline',
                         'Shots on Target Suffered_avg']

# Ranking the values in team_stats_avg
for column in team_stats_ranked.columns:
    if column != 'Team':
        if column in inverted_rank_metrics:
            # Rank in ascending order for inverted ranking metrics
            team_stats_ranked[column] = team_stats_ranked[column].rank(method='min', ascending=True).astype(int)
        else:
            # Rank in descending order for normal ranking metrics
            team_stats_ranked[column] = team_stats_ranked[column].rank(method='min', ascending=False).astype(int)

# Calculate the sum of ranks for each team
rank_columns = [col for col in team_stats_ranked.columns if col != 'Team']
team_stats_ranked['sum_of_ranks'] = team_stats_ranked[rank_columns].sum(axis=1)

# Rank the sum_of_ranks in ascending order
team_stats_ranked['overall_rank'] = team_stats_ranked['sum_of_ranks'].rank(method='min', ascending=True).astype(int)

# Sort by overall rank
team_stats_ranked = team_stats_ranked.sort_values(by='overall_rank', ascending=True).reset_index(drop=True)

# Merge ranks with stats
team_stats_full = pd.merge(team_stats_avg, team_stats_ranked, on='Team', suffixes=('', '_rank')).sort_values(
    by='overall_rank', ascending=True).reset_index(drop=True)

# Create categories for the radar chart
Passing = [
    'Killer Passes_avg_rank',
    'Killer Passes in the Box_avg_rank',
    'Accurate Passes_avg_rank',
    'Accurate Passes in the Box_avg_rank',
    'Key Passes_avg_rank',
    'Passing Accuracy_rank',
    'xA_avg_rank'
]

Finishing = [
    'Goals vs xGoals_rank',
    'no-penalty xG_avg_rank',
    'xGOT_avg_rank',
    'Shots_avg_rank',
    'Shots on Target_avg_rank',
    'Big Chance_avg_rank',
    'Shots OT pct_rank'
]

OnBall = [
    'Ball Recoveries_avg_rank',
    'Ball Recoveries in Offensive Half_avg_rank',
    'High Recoveries pct_rank',
    'Ball Control Ratio_rank',
    'Lost Possessions_avg_rank',
    'Successful Dribbling_avg_rank',
    'Lost Possessions in Zone 1_avg_rank'
]

# Set team colors
team_colors = {
    'Netherlands': 'darkorange',
    'Turkey': 'red',
    'Georgia': 'crimson',
    'Romania': 'gold',
    'Austria': 'firebrick',
    'Spain': 'red',
    'Croatia': 'tomato',
    'Germany': 'black',
    'France': 'royalblue',
    'Poland': 'lightcoral',
    'Ukraine': 'goldenrod',
    'Czech Republic': 'mediumblue',
    'Portugal': 'darkgreen',
    'Switzerland': 'indianred',
    'Hungary': 'forestgreen',
    'Albania': 'maroon',
    'Slovakia': 'dodgerblue',
    'England': 'mediumblue',
    'Denmark': 'indianred',
    'Belgium': 'darkgoldenrod',
    'Scotland': 'navy',
    'Italy': 'mediumblue',
    'Serbia': 'firebrick',
    'Slovenia': 'limegreen'
}

# Store original ranks
original_ranks = team_stats_full.copy()

# Invert rankings
team_stats_full_inverted = team_stats_full.copy()
for col in Passing + Finishing + OnBall:
    team_stats_full_inverted[col] = 25 - team_stats_full_inverted[col]


# Function to prepare data for radar chart
def prepare_radar_data(df, original_df, team_name, metrics_list):
    team_data = df[df['Team'] == team_name][metrics_list + [col.replace('_rank', '') for col in metrics_list]]
    original_team_data = original_df[original_df['Team'] == team_name][metrics_list]
    return team_data, original_team_data


# Function to create a radar chart for one or two teams
def create_radar_chart(team1_name, metrics_list_name, metrics_list, team2_name=None):
    categories = [col.replace('_avg_rank', '').replace('_', ' ') for col in metrics_list]

    # Prepare data for the first team
    team1_data, original_team1_data = prepare_radar_data(team_stats_full_inverted, original_ranks, team1_name,
                                                         metrics_list)
    values1 = team1_data[metrics_list].values.flatten().tolist()
    original_ranks1 = original_team1_data[metrics_list].values.flatten().tolist()
    actuals1 = team1_data[[col.replace('_rank', '') for col in metrics_list]].values.flatten().tolist()
    values1.append(values1[0])  # Repeat the first value to close the circle
    original_ranks1.append(original_ranks1[0])
    actuals1.append(actuals1[0])

    fig = go.Figure()

    # Add trace for the first team
    fig.add_trace(go.Scatterpolar(
        r=values1,
        theta=categories + [categories[0]],
        fill='toself',
        name=f"<b style='color:{team_colors[team1_name]};'>{team1_name}</b>",
        hoverinfo='text',
        text=[f'{cat}: {act}' for cat, orig_rank, act in zip(categories, original_ranks1, actuals1)],
        line=dict(color=team_colors[team1_name], width=3)
    ))

    # Add trace for the second team if provided
    if team2_name:
        team2_data, original_team2_data = prepare_radar_data(team_stats_full_inverted, original_ranks, team2_name,
                                                             metrics_list)
        values2 = team2_data[metrics_list].values.flatten().tolist()
        original_ranks2 = original_team2_data[metrics_list].values.flatten().tolist()
        actuals2 = team2_data[[col.replace('_rank', '') for col in metrics_list]].values.flatten().tolist()
        values2.append(values2[0])  # Repeat the first value to close the circle
        original_ranks2.append(original_ranks2[0])
        actuals2.append(actuals2[0])
        fig.add_trace(go.Scatterpolar(
            r=values2,
            theta=categories + [categories[0]],
            fill='toself',
            name=f"<b style='color:{team_colors[team2_name]};'>{team2_name}</b>",
            hoverinfo='text',
            text=[f'{cat}: {act}' for cat, orig_rank, act in zip(categories, original_ranks2, actuals2)],
            line=dict(color=team_colors[team2_name], width=3)
        ))

    # Update layout for the radar chart
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                showline=False,
                range=[0, 24],
                showticklabels=False  # Remove default ticks
            ),
            angularaxis=dict(
                showline=False,
                showticklabels=False,  # Remove default tick labels
                tickvals=[],
                ticks='',
                linewidth=0
            ),
            bgcolor='rgba(255, 182, 193, 0.12)'
        ),
        showlegend=True,
        title=dict(
            text=f'<b style="color:{team_colors[team1_name]};">{team1_name}</b>' if not team2_name else f'<b style="color:{team_colors[team1_name]};">{team1_name}</b> vs <b style="color:{team_colors[team2_name]};">{team2_name}</b>',
            x=0.5,
            y=0.99,
            xanchor='center',
            yanchor='top',
            font=dict(size=22)
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        annotations=[
            dict(
                x=0.5,
                y=1.12,
                text=f"{metrics_list_name} Rankings",
                showarrow=False,
                font=dict(size=16, color='purple'),
                xanchor='center',
                yanchor='middle'
            )
        ]
    )

    # Set background colors to 'rgba(0,0,0,0)' (transparent)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',  # No background color for the plot area
        paper_bgcolor='rgba(0,0,0,0)',  # No background color for the entire figure
        title=dict(
            font=dict(size=24, color='darkred'),
        ),
    )

    # Add custom labels with manually adjustable positions
    label_positions = [
        {"x": 0.71, "y": 0.5, "text": categories[0], "angle": 90},
        {"x": 0.63, "y": 0.921, "text": categories[1], "angle": 40},
        {"x": 0.45, "y": 1.06, "text": categories[2], "angle": 355},
        {"x": 0.31, "y": 0.75, "text": categories[3], "angle": 305},
        {"x": 0.31, "y": 0.258, "text": categories[4], "angle": 65},
        {"x": 0.46, "y": -0.05, "text": categories[5], "angle": 10},
        {"x": 0.63, "y": 0.05, "text": categories[6], "angle": 320}
    ]

    for label in label_positions:
        fig.add_annotation(
            x=label["x"],
            y=label["y"],
            text=label["text"],
            showarrow=False,
            textangle=label["angle"],
            font=dict(size=15, color='darkblue'),
            xanchor='center',
            yanchor='middle'
        )

    # Add small squares with rank numbers for each metric for the first team
    for i, (rank, value) in enumerate(zip(original_ranks1[:-1], values1[:-1])):  # Skip the repeated first value
        fig.add_trace(go.Scatterpolar(
            r=[value],
            theta=[categories[i]],
            mode='markers+text',
            text=[rank],
            textposition='middle center',
            marker=dict(size=20, color=team_colors[team1_name], symbol='square', line=dict(color='white', width=1)),
            textfont=dict(size=12, color='white'),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Add small squares with rank numbers for each metric for the second team, if provided
    if team2_name:
        for i, (rank, value) in enumerate(zip(original_ranks2[:-1], values2[:-1])):  # Skip the repeated first value
            fig.add_trace(go.Scatterpolar(
                r=[value],
                theta=[categories[i]],
                mode='markers+text',
                text=[rank],
                textposition='middle center',
                marker=dict(size=20, color=team_colors[team2_name], symbol='square', line=dict(color='white', width=1)),
                textfont=dict(size=12, color='white'),
                showlegend=False,
                hoverinfo='skip'
            ))

    # Add My Mark
    fig.add_annotation(
        x=1,
        y=-0.15,
        text='Created by: #DKAnalytics',
        showarrow=False,
        xref='paper',
        yref='paper',
        xanchor='right',
        yanchor='bottom',
        font=dict(size=12, color='#FF4500')
    )

    st.plotly_chart(fig)

### Create Radar Chart

def display_scatter_plot(df, x_metric, y_metric):
    # Replace the metric names with the descriptive names dynamically
    x_metric_display_name = x_metric.replace('_rank', '').replace('_', ' ').replace('avg', '')
    y_metric_display_name = y_metric.replace('_rank', '').replace('_', ' ').replace('avg', '')

    # Calculate mean lines
    x_mean = df[x_metric].mean()
    y_mean = df[y_metric].mean()

    # Create scatter plot
    fig = px.scatter(
        df, x=x_metric, y=y_metric, text='Team',
        color='Team', color_discrete_map=team_colors,
        hover_data={
            x_metric: True,
            y_metric: True,
            'Team': False
        },
        labels={x_metric: x_metric_display_name, y_metric: y_metric_display_name}
        # Use the descriptive names in the labels
    )

    # Update trace and layout
    fig.update_traces(
        textposition='top center',
        marker=dict(size=15, opacity=0.8),
        hovertemplate=(
            '<b>%{text}</b><br>'
            f'{x_metric_display_name}: %{{x}}<br>'
            f'{y_metric_display_name}: %{{y}}<br>'
        )
    )
    fig.update_layout(
        title={
            'text': f"{x_metric_display_name} vs {y_metric_display_name}",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'color': 'darkred', 'size': 24},
        },
        xaxis_title=x_metric_display_name,
        yaxis_title=y_metric_display_name,
        showlegend=False,
        plot_bgcolor='rgba(240, 240, 255, 0.5)',
        xaxis=dict(
            showgrid=False,
            tickfont=dict(color='darkblue'),
            titlefont=dict(color='darkblue'),
            range=[0, None]
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color='darkblue'),
            titlefont=dict(color='darkblue'),
            range=[0, None]
        ),
    )

    # Set background colors to 'rgba(0,0,0,0)' (transparent)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title=dict(
            font=dict(size=24, color='darkred'),
        ),
    )

    # Add mean lines
    fig.add_shape(
        type='line',
        x0=x_mean, y0=0, x1=x_mean, y1=df[y_metric].max(),
        line=dict(color='rgba(255, 0, 0, 0.5)', dash='dash', width=1)  # Thinner, semi-transparent mean line
    )
    fig.add_shape(
        type='line',
        x0=0, y0=y_mean, x1=df[x_metric].max(), y1=y_mean,
        line=dict(color='rgba(255, 0, 0, 0.5)', dash='dash', width=1)  # Thinner, semi-transparent mean line
    )

    # Add my Mark
    fig.add_annotation(
        x=1,
        y=-0.15,
        text='Created by: #DKAnalytics',
        showarrow=False,
        xref='paper',
        yref='paper',
        xanchor='right',
        yanchor='bottom',
        font=dict(size=12, color='#FF4500')
    )

    st.plotly_chart(fig)

### Create Bar Chart

# Function to create and display bar chart for top 10 teams
def display_bar_chart(df, metric):
    # Replace the metric name
    metric_display_name = metric.replace('_rank', '').replace('_', ' ').replace('avg', '')

    # Select only the top 10 teams based on the selected metric
    top_10_teams = df[['Team', metric]].nlargest(10, metric)

    fig = px.bar(
        top_10_teams,
        y='Team',
        x=metric,
        color='Team',
        color_discrete_map=team_colors,
        title=f'Top 10 Teams by {metric}',
        labels={metric: metric_display_name},
        width=800,  # Increase the width of the plot
        height=520  # Increase the height of the plot
    )

    # Set background colors
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title=dict(
            font=dict(size=28, color='darkred'),
        ),
        bargap=0.6,
        barmode='overlay',
        margin=dict(l=50, r=50, t=100, b=50),
    )

    # Update layout to hide labels and center the title
    fig.update_layout(
        title={
            'text': f'Top 10 Teams by {metric_display_name}',
            'x': 0.5,
            'y': 0.88,
            'xanchor': 'center',
            'font': {'color': 'darkred', 'size': 28},
        },
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False,
        yaxis=dict(
            showticklabels=False  # Hide original tick labels
        ),
        xaxis=dict(
            tickfont=dict(
                size=14,
                color='darkblue',
                family='Arial Black'
            )
        ),
        bargap=0.6
    )

    # Customize hover template to show only the metric value
    fig.update_traces(
        hovertemplate='%{x:.1f}<extra></extra>',  # Display the x value formatted to 1 decimal places
        marker=dict(
            line=dict(width=1.5, color='black')
        ),
        hoverlabel=dict(
            font_size=15,  # Increase the font size
            font_family='Arial Black',  # Make the font bold
            font_color='black'
        ),
        text=None
    )

    # Add annotations for y-axis labels
    for i, team in enumerate(top_10_teams['Team']):
        fig.add_annotation(
            x=0,
            y=9 - i,
            text=team,
            showarrow=False,
            font=dict(
                color=team_colors[team],
                size=20  # Increased font size for team names
            ),
            xanchor='right',
            yanchor='middle'
        )

    st.plotly_chart(fig)


# Function to display team rankings card
def display_team_rankings(df, team_name, metrics):
    team_data = df[df['Team'] == team_name][metrics].reset_index(drop=True)
    if team_data.empty:
        st.error("Team not found in the data.")
        return

    # Center and color the title with team color
    team_color = team_colors[team_name] if team_name in team_colors else 'black'
    st.markdown(f"<h3 style='text-align: center; color: {team_color};'>{team_name} Rankings</h3>",
                unsafe_allow_html=True)

    for metric in metrics:
        metric_name = metric.replace('_rank', '').replace('_', ' ')
        rank = team_data.at[0, metric]

        # Calculate the progress value (inverse scale from 24 to 1)
        progress_value = 25 - rank

        # Calculate color based on rank (from red to green) with higher visibility on white background
        red = min(255, int((1 - progress_value / 24) * 255))
        green = min(200, int((progress_value / 24) * 255))
        color = f"rgb({red}, {green}, 0)"

        # Display the metric name and rank with colored progress bar on the same line
        st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="flex: 1; color: darkblue;">
                    <b>{metric_name}:</b>
                </div>
                <div style="flex: 4; position: relative; height: 20px; background-color: #e0e0e0; border-radius: 5px; margin-left: 10px;">
                    <div style="width: {progress_value * 4.16}%; height: 100%; background-color: {color}; border-radius: 5px;"></div>
                </div>
                <div style="flex: 1; text-align: right; font-size: 20px; font-weight: bold; margin-left: 10px; color: {color};">
                    {rank}
                </div>
            </div>
            """, unsafe_allow_html=True)

### Create Passing Networks

def create_team_passing_network(passing_network_data, team_name):
    # Select Team
    team_passing_network_data = passing_network_data[passing_network_data['Team Name'] == team_name]

    # Create average passer locations
    average_locations = team_passing_network_data.groupby('Passer').agg(
        {'Start X': ['mean'], 'Start Y': ['mean', 'count']})
    average_locations.columns = ['x', 'y', 'count']
    average_locations['y'] = 68 - average_locations['y']

    # Calculate passes between players
    passes_between_players = \
    team_passing_network_data.groupby(['Team Name', 'Passer', 'Recipient', 'Player From Name'])[
        'Match ID'].count().reset_index(name='pass_count')

    # Merge passer
    passes_between_players = passes_between_players.merge(average_locations, on='Passer')

    # Merge recipient
    average_locations = average_locations.rename_axis('Recipient').reset_index()
    passes_between_players = passes_between_players.merge(average_locations, on='Recipient', suffixes=['', '_end'])

    return passes_between_players


def plot_team_passing_network(events, players, team_name, match_id):
    # Select the Game
    events = all_events[all_events['Match ID'] == match_id]

    # Filter out extra time
    events = events[events['Period ID'].isin([1, 2])]

    # Find the last Start Minute of Period ID = 1
    last_minute_period_1 = events[events['Period ID'] == 1]['Start Minute'].max()

    # Create the Continuous Minute column
    events['Continuous Minute'] = events.apply(
        lambda row: row['Start Minute'] + last_minute_period_1 if row['Period ID'] == 2 else row['Start Minute'], axis=1
    )

    # Filtering passes from the events data
    passes = events[events['Event Description'] == 'Pass']

    # Function to determine the outcome
    def get_outcome(tags):
        if 'Accurate' in tags:
            return '1'
        elif 'Inaccurate' in tags:
            return '0'
        else:
            return None

    # Function to determine if 'To Penalty Area' is present
    def to_penalty_area(tags):
        return 1 if 'To Penalty Area' in tags else 0

    # Fill NaN values in 'Tags' with empty strings
    passes['Tags'] = passes['Tags'].fillna('')

    # Split the Tags column into lists
    passes['Tags'] = passes['Tags'].str.split(', ')

    # Create the outcome column
    passes['outcome'] = passes['Tags'].apply(get_outcome)

    # Create the 'To Penalty Area' column
    passes['To Penalty Area'] = passes['Tags'].apply(to_penalty_area)

    # Create a new column for accurate passes to penalty area
    passes['Accurate To Penalty Area'] = passes.apply(
        lambda x: 1 if x['outcome'] == '1' and x['To Penalty Area'] == 1 else 0, axis=1)

    # Drop Tags
    passes = passes.drop(columns=['Tags'])

    # Merge events with players to include 'Shirt Number' for Passer
    passes = passes.merge(players[['Player ID', 'Shirt Number']], left_on='Player From ID', right_on='Player ID',
                          how='left')
    passes = passes.rename(columns={'Shirt Number': 'Passer'})

    # Drop the redundant 'Player ID' column from the merge
    passes = passes.drop(columns=['Player ID'])

    # Merge events with players to include 'Shirt Number' for Recipient
    passes = passes.merge(players[['Player ID', 'Shirt Number']], left_on='Player To ID', right_on='Player ID',
                          how='left')
    passes = passes.rename(columns={'Shirt Number': 'Recipient'})

    # Drop the redundant 'Player ID' column from the merge
    passes = passes.drop(columns=['Player ID'])

    # Filter substitutions based on the team
    subs = events[(events['Event Description'] == 'Player Out') & (events['Team Name'] == team_name)]

    # Find the first substitution minute
    first_sub = subs['Continuous Minute'].min()

    # If the first substitution is less than 45, find the next substitution
    if first_sub < 45:
        next_subs = subs[subs['Continuous Minute'] > first_sub]
        if not next_subs.empty:
            first_sub = next_subs['Continuous Minute'].min()
        else:
            first_sub = None  # Handle case where there is no valid substitution after the first one

    # Filter the successful passes before the 1st sub
    successful_passes = passes[passes['outcome'] == '1']
    passing_network_data = successful_passes[successful_passes['Continuous Minute'] <= first_sub]

    # Calculate average pass length
    passing_network_data['Pass Length'] = np.sqrt(
        (passing_network_data['End X'] - passing_network_data['Start X']) ** 2 + (
                    passing_network_data['End Y'] - passing_network_data['Start Y']) ** 2)
    avg_pass_length = passing_network_data['Pass Length'].mean()

    # Convert columns to int64
    passing_network_data['Player From ID'] = passing_network_data['Player From ID'].dropna().astype('int64')
    passing_network_data['Player To ID'] = passing_network_data['Player To ID'].dropna().astype('int64')
    passing_network_data['Recipient'] = passing_network_data['Recipient'].dropna().astype('int64')

    # Create passing network data for the selected team
    team_passing_network = create_team_passing_network(passing_network_data, team_name)
    team_passing_network_data = passing_network_data[passing_network_data['Team Name'] == team_name]

    # Calculate descriptive statistics
    first_sub_minute = team_passing_network_data['Continuous Minute'].max()
    accurate_passes = len(team_passing_network_data)
    forward_passes = len(
        team_passing_network_data[team_passing_network_data['End X'] > team_passing_network_data['Start X']])
    forward_passes_pct = forward_passes / accurate_passes * 100
    passes_final_third = len(team_passing_network_data[(team_passing_network_data['Start X'] < 68) & (
                team_passing_network_data['End X'] > 68)])
    passes_penalty_area = team_passing_network_data[team_passing_network_data['Accurate To Penalty Area'] == 1].shape[0]

    # Create the text for the statistics box
    stats_text = (
        f"Minutes: {first_sub_minute}\n"
        f"Accurate Passes: {accurate_passes}\n"
        f"Fwd Passes: {forward_passes}\n"
        f"Fwd vs Total : {forward_passes_pct:.1f}%\n"
        f"Passes in f3rd: {passes_final_third}\n"
        f"Acc Passes in Box: {passes_penalty_area}\n"
        f"Avg Pass Length: {avg_pass_length:.1f}m"
    )

    # Plot the passing network using mplsoccer
    pitch = Pitch(pitch_type='custom', pitch_color='#f5f5dc', line_color='#696969', pitch_length=105, pitch_width=68)
    fig, ax = pitch.draw(figsize=(10, 7))

    # Define the threshold for pass counts
    pass_threshold = 3

    # Normalize the pass counts for color mapping
    norm = mcolors.Normalize(vmin=3, vmax=10)
    cmap = plt.colormaps['coolwarm']  # selection of the colormap

    # Apply threshold and draw passing lines with gradient colors
    for _, row in team_passing_network[team_passing_network['pass_count'] >= pass_threshold].iterrows():
        color = cmap(norm(row['pass_count']))  # Map the pass count to a color
        pitch.lines(
            row['x'], row['y'],
            row['x_end'], row['y_end'],
            ax=ax, color=color, lw=row['pass_count'] * 0.5, zorder=1
        )

    # Normalize the count values for color mapping
    count_norm = mcolors.Normalize(vmin=team_passing_network['count'].min(), vmax=team_passing_network['count'].max())
    edge_cmap = plt.colormaps['viridis']

    # Add player nodes with colored edges
    nodes = pitch.scatter(
        team_passing_network['x'], team_passing_network['y'],
        s=25 * team_passing_network['count'].values,
        c='#E0FFFF',
        edgecolor=[edge_cmap(count_norm(val)) for val in team_passing_network['count'].values],
        # Map count values to colors
        linewidth=4,
        ax=ax,
        alpha=0.5
    )

    # Add colorbars for the colormaps
    cbar_ax1 = fig.add_axes([0.055, 0.87, 0.2, 0.03])  # [left, bottom, width, height]
    cbar1 = mpl.colorbar.ColorbarBase(cbar_ax1, cmap='coolwarm', norm=norm, orientation='horizontal')
    cbar1.set_label('Passes Between')

    cbar_ax2 = fig.add_axes([0.055, 0.17, 0.2, 0.03])
    cbar2 = mpl.colorbar.ColorbarBase(cbar_ax2, cmap='viridis', norm=count_norm, orientation='horizontal')
    cbar2.set_label('Player Passes')

    # Add player numbers (annotations)
    for i, row in team_passing_network.iterrows():
        pitch.annotate(row['Passer'], xy=(row['x'], row['y']), c='#E65100', fontweight='bold', ha='center', va='center', fontsize=12, ax=ax)

    # Add title annotation
    ax.annotate(f"Passing Network of {team_passing_network['Team Name'].iloc[0]}", xy=(0.5, 0.99), xytext=(0, 0),fontsize=16, xycoords='axes fraction', textcoords='offset points',
                va='top', ha='center', color='#FF4500', fontweight='bold')

    # Add a legend with player numbers and names
    handles = []
    for passer in team_passing_network['Passer'].unique():
        player_name = team_passing_network.loc[team_passing_network['Passer'] == passer, 'Player From Name'].values[0]
        handles.append(plt.Line2D([0], [0], color='w', marker='o', label=f'{passer} - {player_name}',
                                  markerfacecolor='#FF4500', markersize=11, linestyle='None'))

    # Create a FancyBboxPatch for the legend box style
    legend_patch = mpatches.FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.2",
        facecolor='#E0FFFF', edgecolor='#696969', linewidth=2, alpha=0.8
    )

    # Add the legend to the plot with smaller font size
    ax.legend(handles=handles, loc='center left', title=f"{team_name} Players",
              fontsize=10, title_fontsize='10', bbox_to_anchor=(0.81, 0.23),
              fancybox=True, framealpha=0.8, shadow=False, frameon=True)  # Increased transparency with framealpha

    # Manually apply the box style to the legend
    legend = ax.get_legend()
    legend.get_frame().set_facecolor('#E0FFFF')
    legend.get_frame().set_edgecolor('#696969')
    legend.get_frame().set_boxstyle("round,pad=0.2")
    legend.get_frame().set_alpha(0.8)  # Set the transparency

    # Add statistics box
    ax.text(
        0.99, 0.98, stats_text, fontsize=11, va='top', ha='right',
        transform=ax.transAxes,  # Use Axes coordinates
        bbox=dict(facecolor='#E0FFFF', edgecolor='#696969', linewidth=2, boxstyle='round,pad=0.5', alpha=0.8)
    )

    # Add My Mark
    fig.text(0.05, 0.06, 'Created by: #DKAnalytics', ha='left', fontsize=9, color='#FF4500')

    st.pyplot(fig)

### Create Player Spotlight

# The function for plotting player events
def plot_player_events(all_events, player, match_id):
    # Filter events for the specified player and match
    player_events = all_events[(all_events['Player From Name'] == player) & (all_events['Match ID'] == match_id)]

    # Add columns for specific event types and conditions

    # Fw passes
    player_events['fw_passes'] = np.where(
        (player_events['Event Description'].isin(['Pass', 'Killer Pass'])) &
        (player_events['End X'] >= player_events['Start X'] + 5),
        1,
        0
    )

    # Passes to final 3rd
    player_events['to_final_3rd'] = np.where((player_events['Start X'] <= 70) & (player_events['End X'] >= 70), 1, 0)

    player_events['to_final_3rd_accurate'] = np.where(
        (player_events['to_final_3rd'] == 1) & (player_events['Tags'].str.contains('Accurate', na=False)),
        1, 0
    )

    # Pass length
    player_events['Pass Length'] = np.sqrt((player_events['End X'] - player_events['Start X']) ** 2 + (
                player_events['End Y'] - player_events['Start Y']) ** 2).round(2)

    # Define the limits for the right penalty area
    right_penalty_area_x_min = 88.5
    right_penalty_area_x_max = 105
    penalty_area_y_min = 13.85
    penalty_area_y_max = 54.15

    # Create the 'to_penalty_area' column based on the right penalty area
    player_events['to_penalty_area'] = np.where(
        (player_events['End X'] >= right_penalty_area_x_min) & (player_events['End X'] <= right_penalty_area_x_max) &
        (player_events['End Y'] >= penalty_area_y_min) & (player_events['End Y'] <= penalty_area_y_max),
        1, 0
    )

    player_events['accurate_to_area'] = np.where(
        (player_events['to_penalty_area'] == 1) & (player_events['Tags'].str.contains('Accurate', na=False)),
        1, 0
    )

    # Set up the pitch using mplsoccer
    pitch = Pitch(pitch_type='custom', pitch_color='#f5f5dc', line_color='#696969', pitch_length=105, pitch_width=68)
    fig, ax = pitch.draw(figsize=(10, 7))

    # Define shapes and colors for different events
    event_styles = {
        'Duel Won': {'color': 'green', 'marker': 'o', 'size': 100},
        'Intercept': {'color': 'blue', 'marker': 'x', 'size': 100},
        'Duel Lost': {'color': 'red', 'marker': 'o', 'size': 100},
        'Shot': {'color': 'blue', 'marker': 's', 'size': 100},
        'Goal': {'color': 'gold', 'marker': '*', 'size': 500},
    }

    # Plot each event type with different shapes and colors
    for event, style in event_styles.items():
        event_data = player_events[player_events['Event Description'] == event]
        pitch.scatter(event_data['Start X'], 68 - event_data['Start Y'],
                      s=style['size'], color=style['color'], marker=style['marker'], edgecolors='black', ax=ax,
                      label=event)

    # Plot the forward passes
    fw_passes_data = player_events[
        (player_events['fw_passes'] == 1) &
        (player_events['Tags'].str.contains('Accurate', na=False)) &
        (player_events['Pass Length'] > 10)
        ]

    # Plot the forward passes with arrows
    pitch.arrows(fw_passes_data['Start X'], 68 - fw_passes_data['Start Y'],
                 fw_passes_data['End X'], 68 - fw_passes_data['End Y'],
                 color='green', ax=ax, width=1.3, headwidth=3, headlength=4)

    # Add 'o' marker at the start of each pass
    pitch.scatter(fw_passes_data['Start X'], 68 - fw_passes_data['Start Y'],
                  s=100, color='none', marker='o', edgecolors='#FF4500', ax=ax, label='FW Pass Start')

    # Calculate the counts for each event type
    event_counts = {event: len(player_events[player_events['Event Description'] == event]) for event in
                    event_styles.keys()}
    pass_count = len(fw_passes_data)

    # Create a custom legend for the counts
    legend_elements = [
        plt.Line2D([0], [0], marker=style['marker'], color='w', label=f"{event}: {count}",
                  markerfacecolor=style['color'], markersize=10, markeredgewidth=1, markeredgecolor='blue')
        for event, style, count in zip(event_styles.keys(), event_styles.values(), event_counts.values())
    ]

    # Add the pass count to the custom legend
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=f"Progressive Passes: {pass_count}",
                                      markerfacecolor='none', markersize=10, markeredgewidth=1, markeredgecolor='#FF4500'))

    # Display the custom legend as a line at the bottom of the plot
    ax.legend(
        handles=legend_elements,
        loc='upper center',
        fontsize=10,
        ncol=len(legend_elements),
        bbox_to_anchor=(0.4, 0.06),
        fancybox=True,
        frameon=False,
        handletextpad=0.4,    # Reduce the space between the marker and the text
        columnspacing=0.5,    # Reduce the space between columns
        labelspacing=0.4      # Reduce the space between labels
    )

    # Add My Mark
    fig.text(0.98, 0.08, 'Created by: #DKAnalytics', ha='right', fontsize=8.5, color='#FF4500')

    # Add title
    ax.annotate(f"Spotlight on {player_events['Player From Name'].iloc[0]} - {player_events['Game'].iloc[0]}",
                xy=(0.5, 0.99), xytext=(0, 0), fontsize=15, xycoords='axes fraction', textcoords='offset points',
                va='top', ha='center', color='#FF4500', fontweight='bold')

    # Show plot
    st.pyplot(fig)

### Create Shot Maps

# Function to create player shot map
def plot_player_shots(shot_events, player_name, match_id=None):
    # Filter the data based on player name; if match_id is provided, filter further by match_id
    player_data = xg_shot_events[xg_shot_events['Player From Name'] == player_name]
    if match_id is not None:
        player_data = player_data[player_data['Game'] == match_id]

    # Combine Outcome Data
    player_data['Outcome'] = np.where(player_data['Goal'] == 1, 'Goal',
                                      np.where(player_data['GK Save'] == 1, 'GK Save',
                                               np.where(player_data['On Target'] == 1, 'Other', 'Off Target')))

    # Define event styles
    event_styles = {
        'Goal': {'color': 'green', 'marker': 'o'},
        'GK Save': {'color': 'orange', 'marker': 'o'},
        'Other': {'color': 'lightblue', 'marker': '^'},
        'Off Target': {'color': 'red', 'marker': 'x'},
    }

    # Set up the Vertical pitch with padding
    v_pitch = VerticalPitch(half=True, pitch_type='custom', pitch_color='#f5f5dc', line_color='#696969',
                            pitch_length=105, pitch_width=68, pad_bottom=-15, pad_left=-2, pad_right=-2)
    fig, ax = v_pitch.draw(figsize=(8, 6))

    # Plot each event type with different shapes, colors, and marker sizes based on xG
    for event, style in event_styles.items():
        event_data = player_data[player_data['Outcome'] == event]
        v_pitch.scatter(event_data['Start X'], 68 - event_data['Start Y'],
                        s=event_data['xG'] * 1000,  # Adjust marker size based on xG
                        color=style['color'], marker=style['marker'], edgecolors='black', ax=ax, label=event)

    # Create custom legend
    handles = []
    for event, style in event_styles.items():
        handles.append(plt.Line2D([0], [0], marker=style['marker'], color='w',
                                  label=f"{event} ({len(player_data[player_data['Outcome'] == event])})",
                                  markerfacecolor=style['color'], markersize=10, markeredgewidth=1,
                                  markeredgecolor='red'))

    # Add the custom legend
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=4, fontsize=10)

    # Set Title
    ax.set_title(f"Shot Map of {player_name}", fontsize=18, fontweight='bold', color='red', pad=20, y=0.88)

    # Add my mark
    fig.text(0.98, 0.85, 'Creator: #DKAnalytics', ha='right', fontsize=9, color='#FF4500')

    # Show the plot
    st.pyplot(fig)

# Function to create Team shot map
def plot_team_shots(shot_events, team_name, match_id=None):
    # Filter the data based on player name; if match_id is provided, filter further by match_id
    team_data = xg_shot_events[xg_shot_events['Team Name'] == team_name]

    if match_id is not None:
        team_data = team_data[team_data['Game'] == match_id]

    # Combine Outcome Data
    team_data['Outcome'] = np.where(team_data['Goal'] == 1, 'Goal',
                                    np.where(team_data['GK Save'] == 1, 'GK Save',
                                             np.where(team_data['On Target'] == 1, 'Other', 'Off Target')))

    # Define event styles
    event_styles = {
        'Goal': {'color': 'green', 'marker': 'o'},
        'GK Save': {'color': 'orange', 'marker': 'o'},
        'Other': {'color': 'lightblue', 'marker': '^'},
        'Off Target': {'color': 'red', 'marker': 'x'},
    }

    # Set up the Vertical pitch with padding
    v_pitch = VerticalPitch(half=True, pitch_type='custom', pitch_color='#f5f5dc', line_color='#696969',
                            pitch_length=105, pitch_width=68, pad_bottom=-15, pad_left=-2, pad_right=-2)

    fig, ax = v_pitch.draw(figsize=(8, 6))

    # Plot team shots
    for event, style in event_styles.items():
        event_data = team_data[team_data['Outcome'] == event]
        v_pitch.scatter(event_data['Start X'], 68 - event_data['Start Y'],
                        s=event_data['xG'] * 1000,  # Adjust marker size based on xG
                        color=style['color'], marker=style['marker'], edgecolors='black', ax=ax, label=event)

    # Create custom legend
    handles = []
    for event, style in event_styles.items():
        handles.append(plt.Line2D([0], [0], marker=style['marker'], color='w',
                                  label=f"{event} ({len(team_data[team_data['Outcome'] == event])})",
                                  markerfacecolor=style['color'], markersize=10, markeredgewidth=1,
                                  markeredgecolor='red'))

    # Add the custom legend
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.08), ncol=4, fontsize=10)

    # Set Title
    ax.set_title(f"Shot Map of {team_name}", fontsize=18, fontweight='bold', color='red', pad=20, y=0.88)

    # Add my mark
    fig.text(0.98, 0.85, 'Creator: #DKAnalytics', ha='right', fontsize=9, color='#FF4500')

    st.pyplot(fig)

# Function to create Team Conceded shot map
def plot_conceded_team_shots(shot_events, team_name, match_id=None):
    # Filter the data based on player name; if match_id is provided, filter further by match_id
    conceded_data = xg_shot_events[xg_shot_events['opponent team name'] == team_name]
    if match_id is not None:
        conceded_data = conceded_data[conceded_data['Game'] == match_id]

    # Combine Outcome Data
    conceded_data['Outcome'] = np.where(conceded_data['Goal'] == 1, 'Goal',
                                        np.where(conceded_data['GK Save'] == 1, 'GK Save',
                                                 np.where(conceded_data['On Target'] == 1, 'Other', 'Off Target')))

    # Define event styles
    event_styles = {
        'Goal': {'color': 'green', 'marker': 'o'},
        'GK Save': {'color': 'orange', 'marker': 'o'},
        'Other': {'color': 'lightblue', 'marker': '^'},
        'Off Target': {'color': 'red', 'marker': 'x'},
    }

    # Set up the Vertical pitch with padding
    v_pitch = VerticalPitch(half=True, pitch_type='custom', pitch_color='#f5f5dc', line_color='#696969',
                            pitch_length=105, pitch_width=68, pad_bottom=-15, pad_left=-2, pad_right=-2)

    fig, ax = v_pitch.draw(figsize=(8, 6))

    # Plot conceded team shots
    for event, style in event_styles.items():
        event_data = conceded_data[conceded_data['Outcome'] == event]
        v_pitch.scatter(event_data['Start X'], 68 - event_data['Start Y'],
                        s=event_data['xG'] * 1000,  # Adjust marker size based on xG
                        color=style['color'], marker=style['marker'], edgecolors='black', ax=ax, label=event)

    # Create custom legend
    handles = []
    for event, style in event_styles.items():
        handles.append(plt.Line2D([0], [0], marker=style['marker'], color='w',
                                  label=f"{event} ({len(conceded_data[conceded_data['Outcome'] == event])})",
                                  markerfacecolor=style['color'], markersize=10, markeredgewidth=1,
                                  markeredgecolor='red'))

    # Add the custom legend
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.08), ncol=4, fontsize=10)

    # Set Title
    ax.set_title(f"Conceded Shot Map of {team_name}", fontsize=18, fontweight='bold', color='red', pad=20, y=0.88)

    # Add my mark
    fig.text(0.98, 0.85, 'Creator: #DKAnalytics', ha='right', fontsize=9, color='#FF4500')

    st.pyplot(fig)


### APP DESIGN

# Set the page configuration
st.set_page_config(
    page_title="Euro 24 Analytics Dashboard",
    layout="centered",  # Use the full width of the page
    initial_sidebar_state="expanded",  # Expand the sidebar by default
)


def get_image_base64(image_path):
    """Convert image to base64."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


# Load the logos
sdc_logo_path = 'App data/Euro 2024 project/Euro24_Team_Files/Negra verTR.png'
euro24_logo_path = 'App data/Euro 2024 project/Euro24_Team_Files/euro-2024-logo.png'
sics_logo_path = 'App data/Euro 2024 project/Euro24_Team_Files/SICS_Logo.jpeg'
dk_logo_path = 'App data/Euro 2024 project/Euro24_Team_Files/DK_Analytics_logo copy.webp'

# Convert logos to base64
sdc_logo_base64 = get_image_base64(sdc_logo_path)
euro24_base64 = get_image_base64(euro24_logo_path)
sics_base64 = get_image_base64(sics_logo_path)
dk_logo_base64 = get_image_base64(dk_logo_path)

# HTML for displaying logos with links and adjusting sizes
html_logos = f"""
<div style="display: flex; justify-content: center; align-items: center;">
    <a href="https://sportsdatacampus.com/" target="_blank" style="margin-right: 120px;">
        <img src="data:image/png;base64,{sdc_logo_base64}" style="width: 65px; height: auto;">
    </a>
    <a href="https://www.uefa.com/euro2024/" target="_blank" style="margin-right: 120px;">
        <img src="data:image/png;base64,{euro24_base64}" style="width: 65px; height: auto;">
    </a>
    <a href="https://www.sics.it/en/" target="_blank" style="margin-right: 120px;">
        <img src="data:image/png;base64,{sics_base64}" style="width: 125px; height: auto;">
    </a>
    <a href="https://www.linkedin.com/in/dimosthenis-koukias-football-data-analyst/" target="_blank">
        <img src="data:image/png;base64,{dk_logo_base64}" style="width: 70px; height: auto;">
    </a>
</div>
"""

# Display the logos at the top
st.markdown(html_logos, unsafe_allow_html=True)

# Add custom CSS
st.markdown("""
    <style>
        /* Reduce title margins */
        h1, h2, h3 {
            margin-top: 1px;
            margin-bottom: 1px;
        }

        /* Change the background color */
        .stApp {
            background-color: #f0f2f6;
        }

        /* Button styles */
        .stButton > button {
            background-color: blue;
            color: white;
            border-radius: 10px;
            border: 2px solid blue;
            padding: 10px 20px;
        }
        .stButton > button:hover {
            background-color: white;
            color: blue;
        }

        /* Center and resize the logo */
        .logo-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 20px;
        }
        .logo-container img {
            max-width: 150px;
            height: auto;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit app layout
st.markdown("""
    <h1 style='text-align: center; color: darkred;'>Euro 24 Analytics Dashboard</h1>
    <p style='text-align: center; font-size:15px; color: #FF4500;'>by Dimosthenis Koukias</p>
""", unsafe_allow_html=True)

# Add tabs for different sections
tabs = st.tabs(
    ["Passing Networks", "Team Head 2 Head", "Team Comparison", "Top 10", "Rankings Card", "Player Spotlight",
     "Shot Maps"])

# Passing Networks Tab
with tabs[0]:
    # HTML code for hover effect with tooltip and "i" help button
    hover_text = "Select a Team and Game, then press the button to display the Passing Network up to the first substitution (excluding subs in the 1st half). Node size and edge color represent successful passes, while line color and width indicate the number of passes between players (min 3)."
    st.markdown(f"""
          <div style="display: flex; justify-content: center; align-items: center;">
               <h2 style='color: darkblue; margin-right: 10px;'>Passing Networks</h2>
               <div style="position: relative; display: inline-block;">
                   <span style="font-size: 16px; color: darkblue; cursor: pointer;" title="{hover_text}">
                      &#9432;
                   </span>
               </div>
           </div>
       """, unsafe_allow_html=True)

    # Dropdowns for team selection
    col1, col2 = st.columns(2)
    with col1:
        team_name = st.selectbox('Select Team:', options=sorted(all_events['Team Name'].unique()), key='pn_team_name',
                                 help='Select Team')

    with col2:
        filtered_games = all_events[all_events['Team Name'] == team_name]['Game'].unique()
        game = st.selectbox('Select Game:', filtered_games, key='pn_game', help='Select the Game of interest')

    # Button to generate the passing network
    if st.button('Show Passing Network', key='pn_button'):
        match_id = all_events[all_events['Game'] == game]['Match ID'].unique()[0]
        plot_team_passing_network(all_events, players, team_name, match_id)

# Head 2 Head Tab
with tabs[1]:
    hover_text = "Choose teams and the metrics category. Radar shows team rankings per game in the tournament."
    st.markdown(f"""
          <div style="display: flex; justify-content: center; align-items: center;">
               <h2 style='color: darkblue; margin-right: 10px;'>Head To Head</h2>
               <div style="position: relative; display: inline-block;">
                   <span style="font-size: 16px; color: darkblue; cursor: pointer;" title="{hover_text}">
                      &#9432;
                   </span>
               </div>
           </div>
       """, unsafe_allow_html=True)

    # Dropdowns for team selection
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox('Select TeamA:', sorted(team_stats_full['Team'].unique()), key='radar_team1',
                             help='Select the first team')

    with col2:
        team2 = st.selectbox('Select TeamB (optional):', [''] + sorted(list(team_stats_full['Team'].unique())),
                             key='radar_team2', help='Select the second team (optional)')

    # Dropdown for metrics selection
    metrics_option = st.selectbox('Select Category:', ['Passing', 'Finishing', 'OnBall'],
                                  help='Select the metrics category')

    metrics_dict = {
        'Passing': Passing,
        'Finishing': Finishing,
        'OnBall': OnBall
    }

    # Show Radar button
    if st.button('Show Radar', key='radar_button'):
        create_radar_chart(team1, metrics_option, metrics_dict[metrics_option], team2 if team2 else None)

# Team Comparison Tab
with tabs[2]:
    st.markdown("<h2 style='text-align: center; color: darkblue;'>Performance Comparison</h2>", unsafe_allow_html=True)

    # Dropdowns for scatter plot metric selection
    scatter_col1, scatter_col2 = st.columns(2)
    with scatter_col1:
        scatter_x_metric = st.selectbox('Select X-axis Metric:', team_stats_avg.columns[1:], key='scatter_x_metric',
                                        help='Select the metric for the X-axis')
    with scatter_col2:
        scatter_y_metric = st.selectbox('Select Y-axis Metric:', team_stats_avg.columns[1:], key='scatter_y_metric',
                                        help='Select the metric for the Y-axis')

    # Show Scatter Plot button
    if st.button('Show Scatter Plot', key='scatter_button'):
        display_scatter_plot(team_stats_avg, scatter_x_metric, scatter_y_metric)

# Top 10 Tab
with tabs[3]:
    st.markdown("<h2 style='text-align: center; color: darkblue;'>Top 10</h2>", unsafe_allow_html=True)

    # Filter and format metric options
    available_metrics = [col for col in team_stats_avg.columns[1:] if col != 'Goals vs xGoals']
    formatted_metrics = {col: col.replace('_avg', '') for col in available_metrics}

    # Sort the metrics by their formatted names
    sorted_metrics = sorted(formatted_metrics.keys(), key=lambda x: formatted_metrics[x])

    # Dropdown for bar chart metric selection with formatted names
    bar_metric = st.selectbox('Select Metric for Bar Chart:', options=sorted_metrics, format_func=lambda x: formatted_metrics[x], key='bar_metric', help='Select the metric for the bar chart')

    # Show Bar Chart button
    if st.button('Show Bar Chart', key='bar_button'):
        display_bar_chart(team_stats_avg, bar_metric)

# Rankings Card Tab
with tabs[4]:
    st.markdown("<h2 style='text-align: center; color: darkblue;'>Team Rankings Card</h2>", unsafe_allow_html=True)

    # Dropdowns for team selection
    col1, col2 = st.columns(2)
    with col1:
        team_for_card = st.selectbox('Select Team for Rankings Card:', sorted(team_stats_full['Team'].unique()),
                                     key='team_for_card', help='Select the team for the rankings card')

    with col2:
        metrics_for_card = st.selectbox('Select Metrics for Rankings Card:', ['Passing', 'Finishing', 'OnBall'],
                                        help='Select the metrics category for the rankings card')

    # Show Team Rankings button
    if st.button('Show Team Rankings', key='rankings_button'):
        display_team_rankings(team_stats_full, team_for_card, metrics_dict[metrics_for_card])

# Player Spotlight Tab
with tabs[5]:
    # HTML code for hover effect with tooltip and "i" help button
    hover_text = "Select the Team first, the Game and finally the Player to plot and press the button. The graph shows some key actions of the player."
    st.markdown(f"""
        <div style="display: flex; justify-content: center; align-items: center;">
            <h2 style='color: darkblue; margin-right: 10px;'>Player Spotlight</h2>
            <div style="position: relative; display: inline-block;">
                <span style="font-size: 16px; color: darkblue; cursor: pointer;" title="{hover_text}">
                    &#9432;
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Dropdowns for team selection
    col1, col2, col3 = st.columns(3)

    # First, select the team
    with col1:
        team_name = st.selectbox('Select Team:', sorted(all_events['Team Name'].dropna().unique()), key='ps_team_name',
                                 help='Select the Team of interest')

    # Then, select the game, filtered by the selected team
    with col2:
        filtered_games = all_events[all_events['Team Name'] == team_name]['Game'].dropna().unique()
        game = st.selectbox('Select Game:', sorted(filtered_games), key='ps_game', help='Select the Game of interest')

    # Finally, select the player, filtered by the selected team
    with col3:
        filtered_players = all_events[(all_events['Team Name'] == team_name) & (all_events['Game'] == game)][
            'Player From Name'].dropna().unique()
        player_name = st.selectbox('Select Player:', sorted(filtered_players), key='ps_player_name',
                                   help='Select the Player of interest')

    # Button to generate the player spotlight plot
    if st.button('Show Player Spotlight', key='ps_button'):
        match_id = all_events[all_events['Game'] == game]['Match ID'].unique()[0]
        plot_player_events(all_events, player_name, match_id)

# Shot Maps Tab
with tabs[6]:
    # HTML code for hover effect with tooltip and "i" help button
    hover_text = "Choose a team, game, and player. Buttons display shot maps: team shots, conceded shots, or individual player shots. Marker size reflects xG value."
    st.markdown(f"""
          <div style="display: flex; justify-content: center; align-items: center;">
               <h2 style='color: darkblue; margin-right: 10px;'>Shot Maps</h2>
               <div style="position: relative; display: inline-block;">
                   <span style="font-size: 16px; color: darkblue; cursor: pointer;" title="{hover_text}">
                      &#9432;
                   </span>
               </div>
           </div>
       """, unsafe_allow_html=True)

    # Dropdowns
    col1, col2, col3 = st.columns(3)

    with col1:
        # First, select the team
        team_name = st.selectbox('Select Team:', options=sorted(xg_shot_events['Team Name'].unique()))

    with col2:
        # Then, select the game with 'All' option and filter games by selected team
        if team_name:
            filtered_games = ['All'] + sorted(
                xg_shot_events[xg_shot_events['Team Name'] == team_name]['Game'].unique().tolist())
            match_id = st.selectbox('Select Game:', options=filtered_games)

    with col3:
        # Finally, select the player filtered by the selected team
        if team_name:
            filtered_players = xg_shot_events[xg_shot_events['Team Name'] == team_name]['Player From Name'].unique()
            player_name = st.selectbox('Select Player:', options=filtered_players)

    # Buttons to generate the maps
    button_col1, button_col2, button_col3 = st.columns([1, 1, 1])
    show_shot_map = False
    show_conceded_shot_map = False
    show_player_shot_map = False

    with button_col1:
        if st.button('Shot Map', key='shot_map_button'):
            show_shot_map = True

    with button_col2:
        if st.button('Conceded Shot Map', key='conceded_shot_map_button'):
            show_conceded_shot_map = True

    with button_col3:
        if st.button('Player Shot Map', key='player_shot_map_button'):
            show_player_shot_map = True

    # Display the plots in full width without column constraints
    st.write("")  # Adding a separator line

    st.markdown("<div style='padding: 20px;'>", unsafe_allow_html=True)

    # This ensures that the graphs are not constrained by the columns and use the full width of the tab.
    with st.container():
        if show_shot_map:
            plot_team_shots(xg_shot_events, team_name, match_id if match_id != 'All' else None)
        if show_conceded_shot_map:
            plot_conceded_team_shots(xg_shot_events, team_name, match_id if match_id != 'All' else None)
        if show_player_shot_map:
            plot_player_shots(xg_shot_events, player_name, match_id if match_id != 'All' else None)
    st.markdown("</div>", unsafe_allow_html=True)