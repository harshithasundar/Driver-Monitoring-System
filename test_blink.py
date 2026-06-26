from src.analytics.blink_counter import BlinkCounter

counter = BlinkCounter()

samples = [
    0.31,
    0.32,
    0.15,
    0.14,
    0.12,
    0.33,
    0.31,
]

for ear in samples:

    blinks = counter.update(ear)

    print(
        ear,
        blinks
    )