# Euro 24 Analytics Dashboard

## Overview

The **Euro 24 Analytics Dashboard** is a comprehensive web application designed to deliver in-depth analysis and visualizations of the UEFA Euro 2024 tournament. The dashboard allows users to explore various aspects of team and player performance through interactive and visually appealing charts and graphs.

This app is built using Streamlit, a Python framework for creating interactive web applications. It leverages powerful data visualization libraries like Plotly, Matplotlib, and Mplsoccer to provide detailed insights into match events, player actions, and team strategies.

**Data Source:** All data used in this dashboard is provided by [SICS Sports](https://www.sics.it/en/), and the analysis is conducted by [Dimosthenis Koukias](https://www.linkedin.com/in/dimosthenis-koukias-football-data-analyst/).

## Features

### 1. Passing Networks
- **Visualize Passing Networks:** Explore how teams connect passes between players. Node size and edge color indicate the volume and success of passes. The graph includes all passes until the first substitution (if it didn't occur in the first half).
- **Interactive Selection:** Select any team and game to view the passing network up to the first substitution.

### 2. Team Head 2 Head
- **Radar Chart Comparison:** Compare two teams based on selected metrics such as passing, finishing, and on-ball actions. The radar chart displays team rankings, highlighting their relative strengths and weaknesses.
- **Custom Metrics:** Choose from different categories to tailor the radar chart to your analysis needs.

### 3. Team Comparison
- **Scatter Plot Analysis:** Create scatter plots to compare teams based on performance metrics. The plots help in identifying trends, strengths, and weaknesses.
- **Customizable Metrics:** Select any metrics for the X and Y axes to generate the scatter plot.

### 4. Top 10 Teams
- **Bar Chart Rankings:** Visualize the top 10 teams in the tournament based on any selected metric. This feature highlights which teams excel in specific areas.
- **Interactive and Dynamic:** The chart updates dynamically based on the selected metric.

### 5. Rankings Card
- **Detailed Team Rankings:** Get a detailed breakdown of a team's rankings across various metrics. The ranking card includes progress bars that visualize the team's standing.
- **Custom Categories:** Choose from categories like Passing, Finishing, and OnBall to focus the rankings on specific areas.

### 6. Player Spotlight
- **Player Action Visualization:** Analyze the key actions of a selected player during a match. This feature visualizes passes, duels, shots, and other significant events on a football pitch.
- **Interactive and Detailed:** Select the team, game, and player to get a focused view of their actions during the match.

### 7. Shot Maps
- **Shot Visualization:** Explore detailed shot maps for any team or player. The maps include both shots for and against the team, with marker sizes indicating the expected goals (xG) value.
- **Customizable View:** Choose to view team shots, conceded shots, or individual player shots from a specific game or across all games.
