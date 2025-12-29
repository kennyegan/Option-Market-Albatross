from QuantConnect.Orders.Fills import ImmediateFillModel
from datetime import timedelta
import random

class QueueSimFillModel(ImmediateFillModel):
    def __init__(self, algo, latency_ms=50):
        super().__init__()
        self.algo = algo
        self.latency = timedelta(milliseconds=random.gauss(latency_ms, 10))

    def Fill(self, asset, order):
        delayed_time = self.algo.Time + self.latency
        # In production, inject slippage, queueing, partial fills, etc.
        return super().Fill(asset, order)
