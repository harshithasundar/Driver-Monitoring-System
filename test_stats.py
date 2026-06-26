from src.analytics.session_stats import SessionStats

stats = SessionStats()

stats.update_fatigue(20)
stats.update_fatigue(50)
stats.update_fatigue(80)

stats.set_blinks(15)
stats.set_yawns(4)

stats.increment_alerts()
stats.increment_alerts()

print(stats.summary())