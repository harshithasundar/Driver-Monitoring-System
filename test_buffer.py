from src.fatigue.rolling_buffer import RollingBuffer

buffer = RollingBuffer(max_size=5)

for value in [0.1, 0.2, 0.3, 0.4, 0.5]:
    buffer.add(value)

print(buffer.average())

buffer.add(0.9)

print(buffer.average())
print(len(buffer))