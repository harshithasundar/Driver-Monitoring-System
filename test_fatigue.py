from src.fatigue.fatigue_score import FatigueScorer

scorer = FatigueScorer(window_size=5)

samples = [
    0.10,
    0.15,
    0.20,
    0.50,
    0.80,
    0.90,
]

for p in samples:
    score, level = scorer.update(p)

    print(
        f"Probability={p:.2f}"
        f" | Score={score:.1f}"
        f" | {level.value}"
    )