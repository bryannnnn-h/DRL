from preprocessors import *

preprocessed_path = '../0050_done_data_with_ajexid.csv'

file_name = '../tw0050_10.csv'

data = preprocess_data(file_name)
data = add_turbulence(data)
data.to_csv(preprocessed_path)
