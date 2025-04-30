import joblib
from typing import List

from .feature_deal import *
from .utils import *


class Predictor:
    def __init__(self):
        self.base_dir = os.path.dirname(__file__)
        self.model_name = [
    "lgb_sym0_acc0.657.pkl",
    "lgb_sym1_acc0.776.pkl",
    "lgb_sym2_acc0.748.pkl",
    "lgb_sym3_acc0.713.pkl",
    "lgb_sym4_acc0.659.pkl",
    "lgb_sym-1_acc0.706.pkl"
]
        self.else_model = joblib.load
        self.models = [joblib.load(self.base_dir + f"/{name}") for name in self.model_name]

    def pre(self, x_hat, x_columns):
        try:
            # Ensure the "sym" value is within the valid range
            sym_index = int(x_hat["sym"])
            if sym_index < 0 or sym_index >= len(self.models):
                raise ValueError(f"Invalid sym index: {sym_index}")
            return self.models[sym_index].predict(x_hat[x_columns].values.reshape(1, -1))[0]
        except Exception as e:
            return self.models[-1].predict(x_hat[x_columns].values.reshape(1, -1))[0]

    def predict(self, x:List[pd.DataFrame])->List[List[int]]:
        x_columns, sym = main_feature(x[0], True)
        sym = sym if 0 <= sym < 5 else -1
        processed_data = [main_feature(batch.iloc[-1:].copy()).drop("sym", axis=1) for batch in x]
        x_deal = pd.concat(processed_data, ignore_index=True)

        y = self.models[sym].predict(x_deal)
        y = self.generate_signal(y)
        print(y)
        if isinstance(y[0], list):
            return y
        else:
            # 如果模型输出是单维的，包装成List[List[int]]
            return [y]

    def generate_signal(self, data):
        return [[int(i)] for i in data]