"""
Model Comparison Dashboard
Run: streamlit run predictions_app.py
"""
import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Model Comparison", layout="wide")

# Header with refresh button
col_title, col_btn = st.columns([6, 1])
with col_title:
    st.title("🏆 Model Leaderboard")
with col_btn:
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("./results/model_scores.csv")
    df.columns = df.columns.str.strip()  # Strip whitespace from column names
    df['model'] = df['model_name'].apply(lambda x: x.split('/')[-1])
    return df

df = load_data()

# Leaderboard - ranked by Macro F1
st.subheader("📋 Leaderboard (Ranked by Macro F1)")
leaderboard = df[['model', 'model_type', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 
                  'hatespeech_f1', 'normal_f1', 'test_samples']].copy()
leaderboard = leaderboard.rename(columns={'model_type': 'Type'})
leaderboard = leaderboard.sort_values('macro_f1', ascending=False).reset_index(drop=True)
leaderboard.index = leaderboard.index + 1  # 1-indexed ranking
leaderboard.index.name = 'Rank'

# Format as percentages for display
for col in ['accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'hatespeech_f1', 'normal_f1']:
    leaderboard[col] = (leaderboard[col] * 100).round(2)

st.dataframe(leaderboard, use_container_width=True)

# Grouped bar chart for key metrics
st.subheader("📊 Metrics by Model")
metrics_to_plot = ['accuracy', 'macro_f1', 'hatespeech_f1', 'normal_f1']
chart_df = df.melt(id_vars=['model'], value_vars=metrics_to_plot, var_name='Metric', value_name='Score')
chart_df['Score'] = chart_df['Score'] * 100  # Convert to percentage

chart = alt.Chart(chart_df).mark_bar().encode(
    x=alt.X('Metric:N', title=None, axis=alt.Axis(labelAngle=0)),
    y=alt.Y('Score:Q', title='Score (%)', scale=alt.Scale(domain=[0, 100])),
    color=alt.Color('model:N', title='Model'),
    xOffset='model:N'
).properties(height=350)

st.altair_chart(chart, use_container_width=True)

# Per-class performance
st.subheader("🎯 Per-Class Performance")
col1, col2 = st.columns(2)

with col1:
    st.caption("Hatespeech Detection")
    hate_df = df[['model', 'model_type', 'hatespeech_precision', 'hatespeech_recall', 'hatespeech_f1']].copy()
    hate_df.columns = ['Model', 'Type', 'Precision', 'Recall', 'F1']
    for c in ['Precision', 'Recall', 'F1']:
        hate_df[c] = (hate_df[c] * 100).round(2)
    st.dataframe(hate_df.sort_values('F1', ascending=False), hide_index=True, use_container_width=True)

with col2:
    st.caption("Normal Classification")
    norm_df = df[['model', 'model_type', 'normal_precision', 'normal_recall', 'normal_f1']].copy()
    norm_df.columns = ['Model', 'Type', 'Precision', 'Recall', 'F1']
    for c in ['Precision', 'Recall', 'F1']:
        norm_df[c] = (norm_df[c] * 100).round(2)
    st.dataframe(norm_df.sort_values('F1', ascending=False), hide_index=True, use_container_width=True)

# Summary stats
st.subheader("📈 Summary")
best = df.loc[df['macro_f1'].idxmax()]
st.success(f"**Best Model:** {best['model']} with {best['macro_f1']*100:.2f}% Macro F1, {best['accuracy']*100:.2f}% Accuracy")
