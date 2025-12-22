import streamlit as st
import numpy as np
from collections import defaultdict

# === CONFIG ===
EXCLUDE = {4, 5, 28, 52, 69}
NUM_WHITE_BALLS = 69
POSITIONS = 5
BETA_ALPHA_0 = 1.0
BETA_BETA_0 = 68.0

@st.cache_data(ttl=3600)
def generate_data():
    np.random.seed(42)
    draws = []
    for i in range(156):
        white_balls = sorted(np.random.choice(range(1, 70), 5, replace=False))
        draws.append(white_balls)
    return draws

def compute_lifts(draws):
    N = len(draws)
    hits = defaultdict(lambda: defaultdict(int))
    for white_balls in draws:
        for pos, num in enumerate(white_balls, 1):
            hits[num][pos] += 1
    
    results = {}
    for num in range(1, NUM_WHITE_BALLS + 1):
        if num in EXCLUDE: continue
        lifts = []
        for pos in range(1, POSITIONS + 1):
            x = hits[num][pos]
            alpha = BETA_ALPHA_0 + x
            beta = BETA_BETA_0 + N - x
            p_bayes = alpha / (alpha + beta)
            lift = p_bayes * NUM_WHITE_BALLS
            lifts.append(lift)
        results[num] = {'avg_lift': np.mean(lifts)}
    return results

def generate_ticket(top_numbers, previous_draw=None, weighted=True):
    if previous_draw is None: previous_draw = []
    available = [n for n in top_numbers if n not in previous_draw]
    if len(available) < 5: available = top_numbers[:20]
    
    if weighted:
        weights = [results[n]['avg_lift'] for n in available]
        probs = np.array(weights) / sum(weights)
        ticket = np.random.choice(available, 5, replace=False, p=probs)
    else:
        ticket = np.random.choice(available, 5, replace=False)
    return sorted(ticket)

# === APP ===
st.set_page_config(page_title="Powerball Picker", layout="wide")
st.title("🔬 MIT Powerball Picker")

st.sidebar.header("⚙️ Controls")
use_weighted = st.sidebar.checkbox("Weighted picks", True)
prev_input = st.sidebar.text_input("Previous draw", "12,23,45,56,67")
previous_draw = [int(x) for x in prev_input.split(',') if x.isdigit()]

with st.spinner("Computing Bayesian lifts..."):
    draws = generate_data()
    results = compute_lifts(draws)
    ranked = sorted(results.items(), key=lambda x: x[1]['avg_lift'], reverse=True)
    top_15 = [num for num, _ in ranked[:15]]

col1, col2 = st.columns(2)
col1.metric("📊 Draws analyzed", len(draws))
col2.metric("🎯 Top lift", f"{ranked[0][1]['avg_lift']:.3f}")

st.subheader("🏆 Top 15 Numbers (Bayesian Lift)")
for i, (num, data) in enumerate(ranked[:15], 1):
    st.write(f"{i:2d}. **{num:2d}** (lift: {data['avg_lift']:.3f})")

st.subheader("🎫 Your Powerball Tickets")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**🔥 Weighted**")
    ticket1 = generate_ticket(top_15, previous_draw, use_weighted)
    st.success(f"**{list(ticket1)}**")
with col2:
    st.markdown("**🎲 Random**")
    ticket2 = generate_ticket(top_15, previous_draw, False)
    st.success(f"**{list(ticket2)}**")

st.markdown("---")
st.caption("🎓 MIT-style analysis. All lifts ~1.00 = perfectly random. For entertainment only!")
