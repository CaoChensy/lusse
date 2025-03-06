import yaml
import unittest
from lusse.models.manager import register_model_configs
from typing import List, Dict, Union, Optional


class TestYml(unittest.TestCase):
    def test_load_yaml(self):
        with open('../config.yaml', 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        print(data)

    def test_model_manager(self):
        """"""
        with open('../config.yaml', 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        model_managers = register_model_configs(data.get("models"))
        print(len(model_managers))
        print(model_managers["Qwen/Qwen2.5-0.5B-Instruct"])

