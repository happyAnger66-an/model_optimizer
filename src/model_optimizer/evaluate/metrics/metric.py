import abc

class Metric(abc.ABC):
    def __init__(self, result):
        self.result = result

    @abc.abstractmethod
    def print(self):
        raise NotImplementedError(
            "Subclasses must implement the print() method")

    def get_result(self):
        return self.result

    @abc.abstractmethod
    def compare(self, other):
        """
        Compare this metric with another Metric instance.

        Must be implemented by subclasses.

        Args:
            other (Metric): Another Metric instance to compare with.

        Returns:
            Comparison result determined by subclass implementation.
        """
        raise NotImplementedError(
            "Subclasses must implement the compare() method")
