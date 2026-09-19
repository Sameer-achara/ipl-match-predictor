import pickle
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pipe = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))


deliveries_df = pd.read_csv(os.path.join(BASE_DIR, "deliveries.csv"))
matches_df = pd.read_csv(os.path.join(BASE_DIR, "matches.csv"))

st.set_page_config(page_title="IPL Match Predictor",page_icon=os.path.join(BASE_DIR, "ipl.jpg"),layout="wide")
st.title('IPL Win Predictor 🏏')

tab1, tab2 = st.tabs(["🔍 Prediction Model", "📊 Graphical Analysis"])
with tab1:
    col1, col2 = st.columns(2)

    with col1:
     batting_team = st.selectbox('Select Batting Team', sorted(matches_df['team1'].unique().tolist()))
    with col2:
     bowling_team = st.selectbox('Select Bowling Team', sorted(matches_df['team2'].unique().tolist()))

    city = st.selectbox('Select City', sorted(matches_df['city'].dropna().unique().tolist()))
    venue = st.selectbox('Select Venue', sorted(matches_df['venue'].unique().tolist()))

    target = st.number_input('Target Runs', min_value=1, value=100)
    score = st.number_input('Current Score', min_value=0, value=50)
    overs = st.number_input('Overs Completed (e.g. 10.3)', min_value=0.0, max_value=20.0, value=10.0)
    wickets = st.number_input('Wickets Fallen', min_value=0, max_value=10, value=2)

    # Calculations (Jo training ke time logic lagaya tha wahi same yahan lagaya hai)
    runs_left = target - score
    completed_overs = int(overs)
    completed_balls_in_over = round((overs - completed_overs) * 10)
    balls_bowled = (completed_overs * 6) + completed_balls_in_over
    balls_left = 120 - balls_bowled

    # Prediction Button
    if st.button('Predict Win Probability'):

    # Calculations
        runs_left = target - score

        completed_overs = int(overs)
        completed_balls_in_over = round((overs - completed_overs) * 10)

        balls_bowled = (completed_overs * 6) + completed_balls_in_over
        balls_left = 120 - balls_bowled

        wickets_left = 10 - wickets

        # Input for model
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [city],
            'venue':[venue],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_left],
            'target_runs': [target],
        })

        # Prediction
        result = pipe.predict_proba(input_df)[0]

        batting_win = result[1] * 100
        bowling_win = result[0] * 100

        st.success(f"### {batting_team} Win Probability: {round(batting_win, 2)}%")
        st.success(f"### {bowling_team} Win Probability: {round(bowling_win, 2)}%")

with tab2:
   col1,col2 = st.columns(2)
   with col1:
        st.title("Overall IPL analysis")

        st.subheader('Matches Won by Each Team')
        team_wins = matches_df['winner'].value_counts()
        st.bar_chart(team_wins)

        st.subheader("Matches Played in City")
        matches_by_city = matches_df['city'].value_counts()
        st.bar_chart(matches_by_city)

        st.subheader("Toss Decision Analysis")
        toss_decision = matches_df['toss_decision'].value_counts()
        st.bar_chart(toss_decision)

        st.subheader("Toss Winner vs Match Winner")  
        toss_match = (matches_df['toss_winner'] == matches_df['winner']).value_counts()
        toss_match.index = ['Toss Winner Lost', 'Toss Winner Won']
        st.bar_chart(toss_match)


   with col2:
      st.title("Season/Team trends")
      season_matches = matches_df['season'].value_counts().sort_index()
      st.subheader("Season-wise Matches")
      st.line_chart(season_matches)

      season_team_wins = (matches_df.groupby(['season', 'winner']).size().unstack(fill_value=0))
      st.subheader("Most Successful Teams by Season")
      st.bar_chart(season_team_wins)
