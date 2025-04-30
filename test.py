import pandas as pd
from src.utils import *
from need_zip_dir.Predictor import *

base_dir = get_dir_name(True)
data = pd.read_csv(base_dir + "/feature_set/no_feature_sym0.csv")
print("初始化完成")
pre = Predictor()
pre.predict([data[:5], data[5:10]])