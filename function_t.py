import pickle
from re import DEBUG
import numpy as np
import psycopg2
import pandas as pd
from sqlalchemy import create_engine
from Apply import Apply
import utils

import os, json
import pandas as pd
import utils

eng = create_engine("postgresql://princio:postgres@localhost/dns",)
db = psycopg2.connect("host=localhost dbname=dns user=princio password=postgres")

dgas = utils.DGAS0


TN = 0
FP = 1
FN = 2
TP = 3

#
# this function works with more configs
# input:
#       metrics:     array of metrics, one per each config
#       th2s:        [ 0.25, 0.5, 0.75 ]
# output:
#       cms:        [ cm_th0, cm_th1, ..., cm_thn, ]
def function_s2(df, k):
    """
    Input:
        - configs:      array of config
        - metricss:     array of metrics, each corresponding one config
        - k:    the config k that we want to evaluate, fixing the others config ks.

    Output:
        - df:           A Dataframe where the rows correspond to each config less the fixed config k.
                        The columns illustrate the behaviour of this k:
                            - with the mean
                            - with the standard deviation
                            - 
    
    """


    metrics = [ col for col in df.columns if col[0] != 'config' ]

    columns_configs_less_k = [ col for col in df.columns if col[0] == 'config' and col[1] != k ]
    
    configs_less_k = df[columns_configs_less_k].values

    df  = df.sort_values(by=columns_configs_less_k + [('config', k)])
    df  = df.set_index(columns_configs_less_k)

    ks = df[('config', k)].drop_duplicates().to_numpy().tolist()

    metrics_chosen = []
    for dga in utils.DGAS0:
        metrics_chosen += [ (dga, 'int'), (dga, 'int05') ]

    dfs = {}
    for kv in ks:
        _df_ = df[df[('config', k)] == kv].drop(columns=('config', k))

        dfs[kv] = _df_[metrics_chosen].mean()

        if (dfs[kv] >= 1).sum():
            print(dfs[kv])
    

    dfc = pd.DataFrame.from_dict(dfs, orient='index')

    dfc = dfc #/ dfc.mean()
    dfc_n = dfc / dfc.mean()

    dfc = pd.concat([ dfc, dfc.std().rename("std").to_frame().transpose() ], axis = 0)
    dfc = pd.concat([ dfc, dfc.std(axis=1).rename("std").to_frame() ], axis = 1)
    dfc_n = pd.concat([ dfc_n, dfc_n.std().rename("std").to_frame().transpose() ], axis = 0)
    dfc_n = pd.concat([ dfc_n, dfc_n.std(axis=1).rename("std").to_frame() ], axis = 1)

    return dfc, dfc_n





if __name__ == "__main__":
    with open('functions_output_old/configs.json', 'r') as fp:
        configs = json.load(fp)

    hashes = [ hash for hash in configs['configs'] ]

    np.seterr(invalid='ignore')
    
    model2name = utils.model_id2name(db)
    th2 = 0.8
    f1s = []
    for _hash in hashes:
        apply = Apply(_hash)
        f1s.append([
            _hash,
            utils.MODEL_ID2NAME[apply.config.model_id],
            apply.config.top10m,
            apply.config.wsize,
            apply.config.wtype,
            apply.config.inf,
            apply.config.windowing,
            sum(apply.f1score.all > th2) / 200
            ]
        )
        pass

    columns = [
        'hash',
        'model_id',
        'top10m',
        'wsize',
        'wtype',
        'inf',
        'windowing',
        f'f1_{th2}'
    ]

    pd.DataFrame(f1s, columns=columns).sort_values(by=f'f1_{th2}').to_csv(f'f1_{th2}.csv')
    print(pd.DataFrame(f1s, columns=columns).sort_values(by=f'f1_{th2}'))

    pass