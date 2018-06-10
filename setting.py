import os

# Training dataset
data = 'data'
train_data = "data/train"
test_data = "data/test"

# Load model
load_model = None

# set test data's max size
testdata_max_size = None

def dir_init():
    """To init dirs in setting.py """
    params = [data, train_data, test_data]
    for path in params:
        if not os.path.exists(path):
            os.mkdir(path)
    pass
