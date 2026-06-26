from src.analytics.yawn_counter import YawnCounter

counter = YawnCounter()

samples = [
    0.35,
    0.42,
    0.75,
    0.90,
    1.05,
    1.12,
    1.18,
    1.22,
    1.10,
    0.40,
]

for mar in samples:

    yawns = counter.update(mar)

    print(
        mar,
        yawns
    )