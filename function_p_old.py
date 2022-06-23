from re import DEBUG
import psycopg2
import pandas as pd
from function_p import function_o
import utils
from sqlalchemy import create_engine
import numpy as np

eng = create_engine("postgresql://postgres:postgres@localhost/dns",)
db = psycopg2.connect("host=localhost dbname=dns user=postgres password=postgres")

dgas = utils.DGAS0

model_id2name = utils.model_id2name(db)

# input:
#       cms_ths:    result of function l, dataframe of config with a column dedicated to cms, each for th
#       thrange:    [ th0, th1, ..., thn ]
# output:
#       cms:        [ cm_th0, cm_th1, ..., cm_thn, ]
def function_p(cms_ths, th2s):

    p = []
    for i, cm_th_i in enumerate(cms_ths):
        if i %10 == 0:
            print(i)

        metrics = function_o(cm_th_i, th2s)
        
        p.append(metrics)

    return p