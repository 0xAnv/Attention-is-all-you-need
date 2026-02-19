from data import run_data_config_pipeline

data_iterator = run_data_config_pipeline(data_iterator=True)
assert data_iterator is not None, "Data iterator should not be None when data_iterator=True"
print(next(data_iterator).shape)