import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# Load data

events = pd.read_csv('/content/drive/Othercomputers/My Mac/Downloads/Euro24_Team_Files/Match Events/Portugal_vs_Slovenia_events.csv')

shot_events = pd.read_csv('/content/drive/Othercomputers/My Mac/Downloads/Euro24_Team_Files/Match Events/Portugal_vs_Slovenia_shots.csv')

players = pd.read_csv('/content/drive/Othercomputers/My Mac/Downloads/Euro24_Team_Files/all_players-eurocopa-2024.csv')

shots_by_player = pd.read_csv('/content/drive/Othercomputers/My Mac/Downloads/Euro 2024 project/shots_by_player.csv')

stats = pd.read_csv('/content/drive/Othercomputers/My Mac/Downloads/Euro 2024 project/stats/playersStatsSeason-eurocopa-2024.csv')

pd.set_option('display.max_columns', None)

# Initial Transformations

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

team_names = {
    "GEORGIA" : "Georgia",
    "ROMANIA" : "Romania",
    "AUSTRIA" : "Austria",
    "CROAZIA" : "Croatia",
    "GERMANIA" : "Germany",
    "FRANCIA" : "France",
    "POLONIA" : "Poland",
    "UCRAINA" : "Ukraine",
    "REPUBBLICA CECA" : "Czech Republic",
    "PORTOGALLO" : "Portugal",
    "TURCHIA" : "Turkey",
    "SVIZZERA" : "Switzerland",
    "UNGHERIA" : "Hungary",
    "SPAGNA" : "Spain",
    "ALBANIA" : "Albania",
    "SLOVACCHIA" : "Slovakia",
    "OLANDA" : "Netherlands",
    "ITALIA" : "Italy",
    "SERBIA" : "Serbia",
    "SLOVENIA" : "Slovenia",
    "INGHILTERRA" : "England",
    "DANIMARCA" : "Denmark",
    "BELGIO" : "Belgium",
    "SCOZIA" : "Scotland"
}

events['Team Name'] = events['Team Name'].replace(team_names)

# Import glossary to rename columns to english

glossary = pd.read_csv('/content/drive/Othercomputers/My Mac/Downloads/Euro24_Team_Files/glossary-euro-2024.csv')

# Create a dictionary for renaming columns
rename_dict = dict(zip(glossary['Code'], glossary['Description']))

# Rename the columns of player_data
stats.rename(columns=rename_dict, inplace=True)

# Unify player stats with shots aggregates

player_stats = pd.merge(stats, shots_by_player, left_on='Player ID', right_on='Player.id', how='left')

# Calculate games played

player_stats['matches'] = (player_stats['Play Time']/player_stats['Play Time average per game']).fillna(0).astype(int)

# Unify player data with player stats

player_stats = player_stats.merge(players[['Player ID','Shirt Number', 'Position','Position Detail', 'Citizenship', 'Height', 'Foot', 'Photo', 'Age']], on=['Player ID'], how='left')

# Unify positions in one column

player_stats['Position_sm'] = player_stats['Position Detail'].combine_first(player_stats['Position'])

# Create a position dictionary

positions = {'Center Midfield':'CM', 'Centre-Back' : 'CB', 'Striker':'ST', 'Right Winger' : 'Winger', 'Left Winger':'Winger',
             'Midfielder':'CM', 'Goalkeeper':'GK', 'Defender':'CB', 'Defensive Midfield':'CDM', 'Left Forward':'Winger', 'Right-Back':'RB',
             'Attacking Midfield':'CAM', 'Centre-Forward':'ST', 'Left-Back':'LB'}

# Map positions with dictionary for unification

player_stats['Position_sm'] = player_stats['Position_sm'].map(positions)

# Drop unecessary columns

player_stats.drop(columns=['First Name', 'Last Name', 'Born Date', 'Season ID', 'Position', 'Position Detail'], inplace=True)

# Rename columns
player_stats = player_stats.rename(columns={'Citizenship': 'Team','Known Name' : 'Player Name', 'Assists' : 'Key Passes'}) 

# Fix Kosovo Citizenship
player_stats['Team'] = player_stats['Team'].replace('Kosovo','Albania') 

## Create Team aggregated statistics

team_stats = player_stats.groupby(['Team']).agg({
    'Killer Passes'	: 'sum',
    'Killer Passes in the Box' : 'sum',
    'Passes' : 'sum',
    'Accurate Passes' : 'sum',
    'Passes in the Box' : 'sum',
    'Accurate Passes in the Box' : 'sum',
    'Key Passes' : 'sum',
    'Lost Possessions' : 'sum',
    'Ball Recoveries' : 'sum',
    'Ball Recoveries in Offensive Half' : 'sum',
    'Dribbling' : 'sum',
    'Successful Dribbling' : 'sum',
    'no-penalty xG' : 'sum',
    'xGOT' : 'sum',
    'xA' : 'sum',
    'open-play xG' : 'sum',
    'Shots' : 'sum',
    'Shots on Target' : 'sum',
    'Shots from Outside the Box' : 'sum',
    'Crosses' : 'sum',
    'Accurate Crosses' : 'sum',
    'Side Balls' : 'sum',
    'Offsides'  : 'sum',
    'Give and Go' : 'sum',
    'Ball Recoveries in Zone 3' : 'sum',
    'Lost Possessions in Zone 1' : 'sum',
    'Goalkeeper Passes' : 'sum',
    'Successful Goalkeeper Passes' : 'sum',
    'Cut Back' : 'sum',
    'Shots on Target Suffered' : 'sum',
    'Accurate Cut Back' : 'sum',
    'Yellow Cards' : 'sum',
    'Red Cards' : 'sum',
    'matches' : 'max',
    'Goal' : 'sum',
    'Big Chance' : 'sum'
}
).reset_index()

# Creating a new DataFrame for average values
team_stats_avg = team_stats.copy()

# Dividing each column (except 'Team' and 'matches') by the 'matches' column and adding the suffix '_avg'
for column in team_stats.columns:
    if column not in ['Team', 'matches']:
        team_stats_avg[column + '_avg'] = team_stats[column] / team_stats['matches']

# Dropping the original columns to keep only the average columns and 'Team'
team_stats_avg = team_stats_avg[['Team'] + [col + '_avg' for col in team_stats.columns if col not in ['Team', 'matches']]]

# Create success rates

team_stats_avg['Passing Accuracy'] = (team_stats_avg['Accurate Passes_avg'] / team_stats_avg['Passes_avg']) * 100
team_stats_avg['Crossing Accuracy'] = (team_stats_avg['Accurate Crosses_avg'] / team_stats_avg['Crosses_avg']) * 100
team_stats_avg['Passes in Box Accuracy'] = (team_stats_avg['Accurate Passes in the Box_avg'] / team_stats_avg['Passes in the Box_avg']) * 100
team_stats_avg['GK Passing Accuracy'] = (team_stats_avg['Successful Goalkeeper Passes_avg'] / team_stats_avg['Goalkeeper Passes_avg']) * 100
team_stats_avg['Shots OT pct'] = (team_stats_avg['Shots on Target_avg'] / team_stats_avg['Shots_avg']) * 100
team_stats_avg['Dribbling success'] = (team_stats_avg['Successful Dribbling_avg'] / team_stats_avg['Dribbling_avg']) * 100
team_stats_avg['High Recoveries pct'] = (team_stats_avg['Ball Recoveries in Offensive Half_avg'] / team_stats_avg['Ball Recoveries_avg']) * 100
team_stats_avg['Ball Control Ratio'] = (team_stats_avg['Ball Recoveries_avg'] / team_stats_avg['Lost Possessions_avg']) * 100
team_stats_avg['Goals vs xGoals'] = (team_stats_avg['Goal_avg'] - team_stats_avg['no-penalty xG_avg'])
team_stats_avg['Discipline'] = (team_stats_avg['Yellow Cards_avg'] + team_stats_avg['Red Cards_avg'])

# Round values

team_stats_avg = team_stats_avg.round(1)

team_stats_avg = team_stats_avg.drop(columns = ['Passes_avg','Passes in the Box_avg','Dribbling_avg','Crosses_avg','Goalkeeper Passes_avg','Cut Back_avg','Yellow Cards_avg','Red Cards_avg'])

## Rank Teams

# Creating a new DataFrame for ranked values
team_stats_ranked = team_stats_avg.copy()

# List of metrics that should be ranked in ascending order (inverted ranking)
inverted_rank_metrics = ['Lost Possessions_avg','Offsides_avg','Lost Possessions in Zone 1_avg', 'Discipline','Shots on Target Suffered_avg']

# Ranking the values in team_stats_avg
for column in team_stats_ranked.columns:
    if column != 'Team':
        if column in inverted_rank_metrics:
            # Rank in ascending order for inverted ranking metrics
            team_stats_ranked[column] = team_stats_ranked[column].rank(method='min', ascending=True).astype(int)
        else:
            # Rank in descending order for normal ranking metrics
            team_stats_ranked[column] = team_stats_ranked[column].rank(method='min', ascending=False).astype(int)

# Calculating the sum of ranks for each team
rank_columns = [col for col in team_stats_ranked.columns if col != 'Team']
team_stats_ranked['sum_of_ranks'] = team_stats_ranked[rank_columns].sum(axis=1)

# Ranking the sum_of_ranks in ascending order
team_stats_ranked['overall_rank'] = team_stats_ranked['sum_of_ranks'].rank(method='min', ascending=True).astype(int)

# Sort by overall rank

team_stats_ranked = team_stats_ranked.sort_values(by='overall_rank', ascending=True).reset_index(drop=True)

# Merge ranks with stats

team_stats_full = pd.merge(team_stats_avg, team_stats_ranked, on='Team', suffixes=('', '_rank')).sort_values(by='overall_rank', ascending=True).reset_index(drop=True) 

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
    'England': 'red',
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
    team1_data, original_team1_data = prepare_radar_data(team_stats_full_inverted, original_ranks, team1_name, metrics_list)
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
        team2_data, original_team2_data = prepare_radar_data(team_stats_full_inverted, original_ranks, team2_name, metrics_list)
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
            bgcolor='rgba(255, 182, 193, 0.3)'  # Set a more transparent background color
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
            textposition='middle center',  # Position text inside the box
            marker=dict(size=20, color=team_colors[team1_name], symbol='square', line=dict(color='white', width=1)),
            textfont=dict(size=12, color='white'),  # Adjust font size and color for better readability
            showlegend=False
        ))

    # Add small squares with rank numbers for each metric for the second team, if provided
    if team2_name:
        for i, (rank, value) in enumerate(zip(original_ranks2[:-1], values2[:-1])):  # Skip the repeated first value
            fig.add_trace(go.Scatterpolar(
                r=[value],
                theta=[categories[i]],
                mode='markers+text',
                text=[rank],
                textposition='middle center',  # Position text inside the box
                marker=dict(size=20, color=team_colors[team2_name], symbol='square', line=dict(color='white', width=1)),
                textfont=dict(size=12, color='white'),  # Adjust font size and color for better readability
                showlegend=False
            ))

    st.plotly_chart(fig)

# Function to create and display scatter plot
def display_scatter_plot(df, x_metric, y_metric):
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
        }
    )
    
    # Update trace and layout
    fig.update_traces(
        textposition='top center',
        marker=dict(size=15, opacity=0.8),
        hovertemplate=(
            '<b>%{text}</b><br>'
            f'{x_metric}: %{{x}}<br>'
            f'{y_metric}: %{{y}}<br>'
        )
    )
    fig.update_layout(
        title={
            'text': f"{x_metric} vs {y_metric}",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title=x_metric,
        yaxis_title=y_metric,
        plot_bgcolor='rgba(255, 228, 240, 0.1)',  # Lighter pink background
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        showlegend=False
    )
    
    # Add mean lines
    fig.add_shape(
        type='line',
        x0=x_mean, y0=0, x1=x_mean, y1=df[y_metric].max(),
        line=dict(color='red', dash='dash')
    )
    fig.add_shape(
        type='line',
        x0=df[x_metric].min(), y0=y_mean, x1=df[x_metric].max(), y1=y_mean,
        line=dict(color='red', dash='dash')
    )
    
    st.plotly_chart(fig)

# Function to create and display bar chart for top 10 teams
def display_bar_chart(df, metric):
    # Select only the top 10 teams based on the selected metric
    top_10_teams = df[['Team', metric]].nlargest(10, metric)
    
    fig = px.bar(
        top_10_teams,
        y='Team',
        x=metric,
        color='Team',
        color_discrete_map=team_colors,
        title=f'Top 10 Teams by {metric}',
        labels={metric: metric.replace('_rank', '').replace('_', ' ')}
    )
    
    # Update layout to hide labels and center the title
    fig.update_layout(
        title={
            'text': f'Top 10 Teams by {metric}',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'color': 'darkred'}
        },
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False,
        yaxis=dict(
            showticklabels=False  # Hide original tick labels
        ),
        xaxis=dict(
            tickfont=dict(
                size=11,
                color='darkblue',
                family='Arial Black'
            )
    ),
        bargap=0.4
    )
    # Add annotations for y-axis labels with customized colors and sizes
    for i, team in enumerate(top_10_teams['Team']):
        fig.add_annotation(
            x=0,
            y=9-i,
            text=team,
            showarrow=False,
            font=dict(
                color=team_colors[team],
                size=16
            ),
            xanchor='right',
            yanchor='middle'
        )

            # Add creator's mark
    fig.add_annotation(
        x=1,
        y=-0.15,
        text='#DKAnalytics',
        showarrow=False,
        xref='paper',
        yref='paper',
        xanchor='right',
        yanchor='bottom',
        font=dict(size=12, color='grey')
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
    st.markdown(f"<h3 style='text-align: center; color: {team_color};'>{team_name} Rankings</h3>", unsafe_allow_html=True)
    
    for metric in metrics:
        metric_name = metric.replace('_rank', '').replace('_', ' ')
        rank = team_data.at[0, metric]

        # Calculate the progress value (inverse scale from 24 to 1)
        progress_value = 25 - rank
        
        # Calculate color based on rank (from red to green)
        color_step = int((progress_value / 23) * 255)
        color = f"rgb({255 - color_step}, {color_step}, 0)"
        
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

# Streamlit app layout
st.markdown("<h1 style='text-align: center; color: darkred;'>Team Performance Charts</h1>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: darkblue;'>Radar Chart</h2>", unsafe_allow_html=True)

# Dropdowns for team selection with styling
col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox('Select TeamA:', team_stats_full['Team'].unique(), key='team1', help='Select the first team')

with col2:
    team2 = st.selectbox('Select TeamB (optional):', [''] + list(team_stats_full['Team'].unique()), key='team2', help='Select the second team (optional)')

# Dropdown for metrics selection with styling
metrics_option = st.selectbox('Select Metrics:', ['Passing', 'Finishing', 'OnBall'], help='Select the metric to display')

metrics_dict = {
    'Passing': Passing,
    'Finishing': Finishing,
    'OnBall': OnBall
}

# Show Radar button with styling
if st.button('Show Radar', help='Click to display the radar chart'):
    create_radar_chart(team1, metrics_option, metrics_dict[metrics_option], team2 if team2 else None)

st.markdown("<h2 style='text-align: center; color: darkblue;'>Scatter Plot</h2>", unsafe_allow_html=True)

# Dropdowns for scatter plot metric selection with styling
scatter_col1, scatter_col2 = st.columns(2)

with scatter_col1:
    scatter_x_metric = st.selectbox('Select X-axis Metric:', team_stats_avg.columns[1:], key='scatter_x_metric', help='Select the metric for the X-axis')

with scatter_col2:
    scatter_y_metric = st.selectbox('Select Y-axis Metric:', team_stats_avg.columns[1:], key='scatter_y_metric', help='Select the metric for the Y-axis')

# Show Scatter Plot button with styling
if st.button('Show Scatter Plot', help='Click to display the scatter plot'):
    display_scatter_plot(team_stats_avg, scatter_x_metric, scatter_y_metric)

st.markdown("<h2 style='text-align: center; color: darkblue;'>Bar Chart</h2>", unsafe_allow_html=True)

# Dropdown for bar chart metric selection
bar_metric = st.selectbox('Select Metric for Bar Chart:', [col for col in team_stats_avg.columns[1:]], key='bar_metric', help='Select the metric for the bar chart')

# Show Bar Chart button 
if st.button('Show Bar Chart', help='Click to display the bar chart'):
    display_bar_chart(team_stats_avg, bar_metric)

st.markdown("<h2 style='text-align: center; color: darkblue;'>Team Rankings Card</h2>", unsafe_allow_html=True)

# Dropdown for team selection and metrics for the rankings card
team_for_card = st.selectbox('Select Team for Rankings Card:', team_stats_full['Team'].unique(), key='team_for_card', help='Select the team for the rankings card')
metrics_for_card = st.selectbox('Select Metrics for Rankings Card:', ['Passing', 'Finishing', 'OnBall'], help='Select the metrics category for the rankings card')

# Show Team Rankings button with styling
if st.button('Show Team Rankings', help='Click to display the team rankings'):
    display_team_rankings(team_stats_full, team_for_card, metrics_dict[metrics_for_card])
