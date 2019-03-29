import os
from config import Config


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

con = Config()
con.load_testt_data()
con.set_testt_model()
con.test()