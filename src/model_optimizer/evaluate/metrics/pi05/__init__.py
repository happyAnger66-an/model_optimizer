from model_optimizer.evaluate.metrics.metric import Metric
from model_optimizer.evaluate.compare.utils import compare_predictions

class Pi05Metric(Metric):
    def __init__(self, result):
        super().__init__(result)

    def compare(self, other: Metric):
        other_result = other.get_result()
        self_result = self.get_result()
        for origin_result, quantized_result in zip(self_result, other_result):
            compare_predictions(origin_result, quantized_result, key1="Origin", key2="Quantized")

    def print(self):
        pass