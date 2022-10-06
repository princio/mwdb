
from dataclasses import dataclass
import hashlib
import json
import os
import pickle
from typing import Tuple, Union
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

TPVector = npt.NDArray[Union[np.int_, np.float64]]

CMType = list[ tuple[int, int, tuple[int, int, int, int], tuple[int, int, int, int] ] ]

TPVectorType = tuple[ TPVector, TPVector, TPVector ]

CMVectorizedType = tuple[ TPVector, TPVector, TPVectorType, TPVectorType ]

class MetrixDict(TypedDict):
    n: TPVector
    all: TPVector
    dga_1: TPVector
    dga_2: TPVector

class PositivesDict(TypedDict):
    all: TPVector
    dga_1: TPVector
    dga_2: TPVector

class Positives:
    all: TPVector
    dga_1: TPVector
    dga_2: TPVector

    def __init__(self, positives: TPVectorType) -> None:
        self.all = positives[0]
        self.dga_1 = positives[1]
        self.dga_2 = positives[2]
        pass

    def __getitem__(self, index) -> TPVector:
        if (type(index) == str):
            return { 'all': self.all, 'dga_1': self.dga_1, 'dga_2': self.dga_2}[index]
        return [self.all, self.dga_1, self.dga_2][index]
    
    @classmethod
    def init2(cls, all: TPVector, dga_1: TPVector, dga_2: TPVector):
        return cls((all, dga_1, dga_2))

    @classmethod
    def from_dict(cls, dict_: PositivesDict):
        return cls((dict_['all'], dict_['dga_1'], dict_['dga_2']))

    pass

class Metric:
    def __init__(self, n: TPVector, all: TPVector, dga_1: TPVector, dga_2: TPVector) -> None:
        self.n: TPVector = n
        self.p: Positives = Positives.init2(all, dga_1, dga_2)
    
    @classmethod
    def from_dict(cls, _d: MetrixDict):
        return cls(_d['n'], _d['all'], _d['dga_1'], _d['dga_2'])

class ConfusionMatrix:
    def __init__(self, cm: CMVectorizedType):
        self.data = []
        self.tn = cm[0]
        self.fp = cm[1]
        self.fn = Positives(cm[2])
        self.tp = Positives(cm[3])


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
            self.cms = ConfusionMatrix(typing.cast(CMVectorizedType, pickle.load(f)))

        pass

    @property
    def tnr(self) -> TPVector:
        return self.cms.tn / (self.cms.tn + self.cms.fp)

    @property
    def recall(self) -> Positives:
        d = {}
        for dga in ['all', 'dga_1', 'dga_2']:
            d[dga] = self.cms.tp[dga] / (self.cms.tp[dga] + self.cms.fn[dga])
        return Positives.from_dict(typing.cast(PositivesDict, d))
    
    @property
    def precision(self) -> Positives:
        d = {}
        for dga in ['all', 'dga_1', 'dga_2']:
            d[dga] = self.cms.tp[dga] / (self.cms.tp[dga] + self.cms.fp)

        return Positives.from_dict(typing.cast(PositivesDict, d))
    
    @property
    def accuracy(self) -> Positives:
        d = {}
        for dga in ['all', 'dga_1', 'dga_2']:
            d[dga] = self.cms.tp[dga] / ((self.cms.tp[dga] + self.cms.fn[dga]) + self.cms.fp)

        return Positives.from_dict(typing.cast(PositivesDict, d))

    
    @property
    def f1score(self) -> Positives:
        pr = self.precision
        re = self.recall
        
        f1: PrecisionsMetricType = {}  # type: ignore
        for a in [ 'all', 'dga_1', 'dga_2']:
            f1[a] = (2 * pr[a] * re[a]) / (pr[a] + re[a])
        
        return Positives.from_dict(typing.cast(PositivesDict, f1))