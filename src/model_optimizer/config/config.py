import json
import addict

class Config:
    def __init__(self, filepath):
        self.filepath = filepath
        with open(filepath, 'r') as f:
            self.config = addict.Dict(json.load(f))