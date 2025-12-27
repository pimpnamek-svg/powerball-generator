from fastapi import FastAPI
import random
import time
import os
import hashlib
import numpy as np
from collections import defaultdict

app = FastAPI()

@app.on_event("startup")
def reset_rng_on_startup():
    entropy = f"{time.time()}-{os.getpid()}-{os.urandom(16)}"
    seed = int(hashlib.sha256(entropy.encode()).hexdigest(), 16)
    random.seed(seed)
    print("🎲 RNG seeded on startup")


app = FastAPI(title="Powerball Picker API")

# === CONFIG ===
EXCLUDE = {4, 5, 28, 52, 69}
NUM_WHITE_BALLS = 69
POSITIONS = 5
BETA_ALPHA_0 = 1.0
BETA_BETA_0 = 68.0


def generate_data():
    draws = []
    for _ in range(156):
        white_balls = sorted(
            np.random.choice(range(1, 70), 5, replace=False)
        )
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
        if num in EXCLUDE:
            continue

        lifts = []
        for pos in range(1, POSITIONS + 1):
            x = hits[num][pos]
            alpha = BETA_ALPHA_0 + x
            beta = BETA_BETA_0 + N - x
            p_bayes = alpha / (alpha + beta)
            lift = p_bayes * NUM_WHITE_BALLS
            lifts.append(lift)

        results[num] = {"avg_lift": float(np.mean(lifts))}

    return results


def generate_ticket(top_numbers, results, previous_draw=None, weighted=True):
    if previous_draw is None:
        previous_draw = []

    available = [n for n in top_numbers if n not in previous_draw]
    if len(available) < 5:
        available = top_numbers[:20]

    if weighted:
        weights = np.array([results[n]["avg_lift"] for n in available])
        probs = weights / weights.sum()
        ticket = np.random.choice(available, 5, replace=False, p=probs)
    else:
        ticket = np.random.choice(available, 5, replace=False)

    return sorted(ticket.tolist())


# === ROUTES ===

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Powerball API running"}


@app.get("/generate")
def generate(previous_draw: str = "", weighted: bool = True):
    prev = [int(x) for x in previous_draw.split(",") if x.isdigit()]

    draws = generate_data()
    results = compute_lifts(draws)

    ranked = sorted(results.items(), key=lambda x: x[1]["avg_lift"], reverse=True)
    top_15 = [num for num, _ in ranked[:15]]

    ticket_weighted = generate_ticket(top_15, results, prev, weighted)
    ticket_random = generate_ticket(top_15, results, prev, False)

    return {
        "draws_analyzed": len(draws),
        "top_15": top_15,
        "weighted_ticket": ticket_weighted,
        "random_ticket": ticket_random,
    }

