from function_f import function_f
from function_l import function_l
from function_p import function_p
from function_s import function_s
from function_s2 import function_s2
from utils import ApplyConfiguration, load

import os, json
import pandas as pd
import utils



if __name__ == "__main__":
    with open('functions_output/configs.json', 'r') as fp:
        configs = json.load(fp)
    nth = configs['nth']
    th2s = configs['th2s']
    configs = [
        ApplyConfiguration.from_dict(configs['configs'][hash])
        for hash in configs['configs']
    ]

    metrics_path = './functions_output/metrics.csv'
    redo = True
    if not redo and os.path.exists(metrics_path):
        df_metrics = pd.read_csv(metrics_path, header=[0,1], index_col=None)
    else:
        metrics = []
        for i, config in enumerate(configs):
            config_hash = config.__hash__()

            df_f, cms_ths, ths = load(config_hash)

            p = function_p(cms_ths, ths)

            p = { (dga, m): p[dga][m] for dga in utils.DGAS0 for m in [ 'int', 'int05' ] }

            metrics.append(p)

            pass

        df_metrics = pd.DataFrame(metrics)
        df_metrics.columns = pd.MultiIndex.from_tuples(df_metrics)
        df_metrics.to_csv(metrics_path, index=None)

        pass
    
    for i, metric in df_metrics.iterrows():
        config_hash = configs[i].__hash__()
        if any([ metric[(dga, 'int')] > 1 for dga in utils.DGAS0 ]):
            print([ metric[(dga, 'int')] > 1 for dga in utils.DGAS0 ])
            df_f, cms_ths, ths = load(config_hash)
            function_p(cms_ths, ths, plot=True)

    # merge with configs
    df_configs = pd.DataFrame([{ ('config', k) : v  for k,v in config.to_dict().items() } for config in configs ])
    df_configs.columns = pd.MultiIndex.from_tuples(df_configs.columns)
    df = pd.concat([df_configs, df_metrics.reset_index(drop=True)], axis=1)

    for fixed_metric in [ 'model_id', 'top10m', 'wsize', 'windowing', 'inf' ]:

        dfc, dfc_n = function_s2(df, fixed_metric)

        print(dfc.round(2).to_markdown())
        print(dfc_n.round(2).to_markdown())

        pass

    pass