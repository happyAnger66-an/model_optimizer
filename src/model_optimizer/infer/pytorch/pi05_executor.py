from ..executor import Executor

class Pi05PyTorchExecutor(Executor):
    def __init__(self, policy):
        super().__init__(policy)
        self.policy = policy

    def load_model(self, config=None):
        pass