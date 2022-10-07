
from dataclasses import dataclass
import hashlib
import json
import os
import pickle
from typing import Generic, NamedTuple, Tuple, TypeVar, Union
import typing
import numpy as np
import numpy.typing as npt
from typing import TypedDict

import pandas as pd

ROOT_DIR = '.'

TN = 0
FP = 1
FN = 2
TP = 3


TPVector = npt.NDArray[np.int_]

MetricVector = npt.NDArray[np.float64]

IntDatum = np.int_

Datum = Union[IntDatum, TPVector]

MetricDatum = Union[np.float64, MetricVector]

CMType = list[ tuple[IntDatum, IntDatum, tuple[IntDatum, IntDatum, IntDatum, IntDatum], tuple[IntDatum, IntDatum, IntDatum, IntDatum] ] ]

TPVectorType = tuple[ TPVector, TPVector, TPVector ]

CMVectorizedType = tuple[ TPVector, TPVector, TPVectorType, TPVectorType ]

T = TypeVar('T', IntDatum, TPVector)
K = TypeVar('K', np.float64, MetricVector)


class Value2DGA(Generic[T]):
    def __init__(self, values: tuple[T, T, T]):
        self.values = values
        pass

    def __getitem__(self, key):
        if type(key) == str:
            index = { 'all': 0, 'dga_1': 1, 'dga_2': 2}[typing.cast(str, key)]
        else:
            index = typing.cast(int, key)
        return self.values[index]
    
    @classmethod
    def init2(cls, all: T, dga_1: T, dga_2: T):
        return cls((all, dga_1, dga_2))
    
    def __str__(self):
        print(self.values[0])
        return f'( {self.values[0]}, {self.values[1]}, {self.values[2]} )'

    pass

# class Metric(Generic[K]):
#     def __init__(self, data: tuple[ T, T, T ]) -> None:
#         self.p: Value2DGA[T] = Value2DGA[T].init2(data[0], data[1], data[2])

#     def __getitem__(self, index):
#         return self.p[index]
    
#     def __str__(self):
#         return self.p.__str__()

class Metric(Generic[K]):
    def __init__(self, values: tuple[T, T, T]):
        self.values = values
        pass

    def __getitem__(self, key):
        if type(key) == str:
            index = { 'all': 0, 'dga_1': 1, 'dga_2': 2}[typing.cast(str, key)]
        else:
            index = typing.cast(int, key)
        return self.values[index]
    
    @classmethod
    def init2(cls, all: T, dga_1: T, dga_2: T):
        return cls((all, dga_1, dga_2))
    
    def __str__(self):
        return f'( {self.values[0]}, {self.values[1]}, {self.values[2]} )'

    pass
    

class ConfusionMatrix(Generic[T]):
    def __init__(self, cm: tuple[ T, T, Value2DGA[T], Value2DGA[T]]):
        self.tn = cm[0]
        self.fp = cm[1]
        self.fn = cm[2]
        self.tp = cm[3]
        pass

    def __str__(self):
        return f'{self.tn} {self.fp} {self.fn} {self.tp}'

    @property
    def tnr(self) -> MetricDatum:
        return np.round(self.tn / (self.tn + self.fp), 2)

    @property
    def recall(self) -> Metric[K]:
        v = []
        for dga in [ 0, 1, 2 ]:
            v.append(self.tp[dga] / (self.tp[dga] + self.fn[dga]))
            v[dga] = np.round(v[dga], 2)
        return Metric[K](tuple(v))
    
    @property
    def precision(self) -> Metric[K]:
        v = [ ]
        for dga in [ 0, 1, 2 ]:
            v.append(self.tp[dga] / (self.tp[dga] + self.fp))
            v[dga] = np.round(v[dga], 2)
        return Metric[K](tuple(v))
    
    @property
    def accuracy(self) -> Metric[K]:
        v = [ ]
        for dga in [ 0, 1, 2 ]:
            den = ((self.tp[dga] + self.fn[dga]) + self.fp)
            v.append(self.tp[dga] / den)
            v[dga] = np.round(v[dga], 2)
        return Metric[K](tuple(v))

    
    @property
    def f1score(self) -> Metric[K]:
        pr = self.precision
        re = self.recall
        
        v = [ ]
        for dga in [ 0, 1, 2 ]:
            den = (pr[dga] + re[dga])
            v.append((2 * pr[dga] * re[dga]) / den)
            v[dga] = np.round(v[dga], 2)
        
        return Metric[K](tuple(v))

    def metric(self, metric_name) -> Metric[K]:
        return getattr(self, metric_name)

    pass

@dataclass
class ApplyConfiguration:
    name: str
    model_id: str
    wtype: str
    top10m: Tuple[bool, int]
    wsize: int
    windowing: str
    inf: Tuple[int, int]
    pass

    @staticmethod
    def from_dict(dict):
        return ApplyConfiguration(
            name=dict['name'] if 'name' in dict else None,  # type: ignore
            model_id=dict['model_id'],
            wtype=dict['wtype'],
            top10m=tuple(dict['top10m']),
            wsize=dict['wsize'],
            windowing=dict['windowing'],
            inf=tuple(dict['inf'])
        )
    
    def to_dict(self):
        d = dict(
            name=self.name,
            model_id=self.model_id,
            wtype=self.wtype,
            top10m=self.top10m,
            wsize=self.wsize,
            windowing=self.windowing,
            inf=self.inf
        )
        if self.wtype == 'nx':
            d['inf'] = ( -20, 20 )
            d['model_id'] = 'nx %0.1f, %0.1f' % self.inf

        return d

    def __hash__(self):
        sha = hashlib.md5()
        seed = repr((
            self.model_id,
            self.wtype,
            self.top10m,
            self.wsize,
            self.windowing,
            self.inf
        )).encode('utf-8')
        sha.update(seed)
        return sha.hexdigest()
    
    def tolist(self):
        return [ self.model_id, self.wtype, self.top10m, self.wsize, self.windowing, self.inf ]


class Apply:
    def __init__(self, hash: str) -> None:
        self.hash = hash

        with open('functions_output_old/configs.json', 'r') as fp:
            configs = json.load(fp)
        self.config: ApplyConfiguration = ApplyConfiguration.from_dict(configs['configs'][hash])

        # windows: index,wlen,wvalue,wnum,dga,pcap_id
        self.windows = pd.read_csv(os.path.join(ROOT_DIR, f'./functions_output_2dga/f/{hash}.2dga.csv'), index_col=0)

        self.ths = np.linspace(self.windows.wvalue.min(), self.windows.wvalue.max(), num=configs['nth'], dtype=np.float32)

        _path_pkl = os.path.join(ROOT_DIR, f'./functions_output_2dga/f/{hash}.vectorized.pickle')
        with open(_path_pkl, 'rb') as f:
            self.cm = ConfusionMatrix[TPVector](pickle.load(f))

        pass

    def cm_th(self, th):
        df = self.windows
        wnum = {
            'legit': sum(df.dga == 0),
            'all': sum(df.dga > 0),
            'dga_1': sum(df.dga == 1),
            'dga_2': sum(df.dga == 2),
        }

        tn = np.int_(sum((df.dga == 0) & (df.wvalue <= th)))
        fp = np.int_(wnum['legit'] - tn)

        df_tp = df[(df.dga > 0) & (df.wvalue > th)]

        tp = np.zeros(3, dtype=np.int_)
        fn = np.zeros(3, dtype=np.int_)
        for vdga, dga in enumerate([ 'all', 'dga_1', 'dga_2' ]):
            if dga == 0:
                tp[vdga] = np.int_(df_tp.shape[0])
                fn[vdga] = np.int_(wnum[dga] - df_tp.shape[0])
            else:
                tp[vdga] = np.int_(sum(df_tp.dga == vdga))
                fn[vdga] = np.int_(wnum[dga] - tp[vdga])
            pass

        return ConfusionMatrix[np.int_](( tn, fp, Value2DGA[np.int_](tuple(fn)), Value2DGA[np.int_](tuple(tp)) ))