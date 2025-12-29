from AlgorithmImports import *
from quantconnect.queue_sim_fill_model import QueueSimFillModel

class OMAAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2023, 3, 1)
        self.SetCash(1000000)

        option = self.AddOption("SPY")
        option.SetFilter(-2, 2, timedelta(0), timedelta(30))

        self.SetExecution(ImmediateExecutionModel())
        self.SetFillModel(QueueSimFillModel(self))

    def OnData(self, slice: Slice):
        if not self.Portfolio.Invested and slice.OptionChains:
            for chain in slice.OptionChains:
                contracts = [x for x in chain.Value if x.Right == OptionRight.Call]
                if contracts:
                    self.MarketOrder(contracts[0].Symbol, 1)
