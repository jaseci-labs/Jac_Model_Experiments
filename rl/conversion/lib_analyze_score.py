import os
# AnalyticsReport(pid: int, score: float, service: str = "analytics") obj provided elsewhere.
def analyze(signatures: int) -> "AnalyticsReport":
    return AnalyticsReport(pid=os.getpid(), score=float(signatures) * 1.5)
