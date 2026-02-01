class Executor:
    def __init__(self, policy):
        self.policy = policy

    def load_model(self):
        pass

    def __getattr__(self, name):
        return getattr(self.policy, name)