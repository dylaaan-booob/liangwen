import pandas as pd
from src.utils import get_dir_name
base_dir = get_dir_name(True)
data = pd.read_csv(base_dir + "/feature_set/no_feature_sym0.csv")

