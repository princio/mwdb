
from dataclasses import dataclass
import hashlib
import json
import os
import pickle
from typing import List, Tuple, Union
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

ConfusionMatrixType = Tuple[ List[int], List[int], Tuple[ List[int], List[int], List[int] ], Tuple[ List[int], List[int], List[int] ] ] 

TPVector = npt.NDArray[Union[np.int_, np.float64]]


CMDGA2Type = list[ tuple[int, int, tuple[int, int, int, int], tuple[int, int, int, int] ] ]
CMDGA3Type = list[ tuple[int, int, tuple[int, int, int], tuple[int, int, int] ] ]

TPVectorDGA3Type = tuple[ TPVector, TPVector, TPVector, TPVector ]
TPVectorDGA2Type = tuple[ TPVector, TPVector, TPVector ]

CMVectorizedDGA3Type = tuple[ TPVector, TPVector, TPVectorDGA3Type, TPVectorDGA3Type ]
CMVectorizedDGA2Type = tuple[ TPVector, TPVector, TPVectorDGA2Type, TPVectorDGA2Type ]

class MetrixDict(TypedDict):
    n: TPVector
    all: TPVector
    dga_1: TPVector
    dga_2: TPVector

class PositivesDGA2Dict(TypedDict):
    all: TPVector
    dga_1: TPVector
    dga_2: TPVector

class PositivesDGA2:
    all: TPVector
    dga_1: TPVector
    dga_2: TPVector

    def __init__(self, positives: TPVectorDGA2Type) -> None:
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
    def from_dict(cls, dict_: PositivesDGA2Dict):
        return cls((dict_['all'], dict_['dga_1'], dict_['dga_2']))

    pass

class Metric:
    def __init__(self, n: TPVector, all: TPVector, dga_1: TPVector, dga_2: TPVector) -> None:
        self.n: TPVector = n
        self.p: PositivesDGA2 = PositivesDGA2.init2(all, dga_1, dga_2)
    
    @classmethod
    def from_dict(cls, _d: MetrixDict):
        return cls(_d['n'], _d['all'], _d['dga_1'], _d['dga_2'])

class ConfusionMatrix:
    def __init__(self, cm: CMVectorizedDGA2Type):
        self.data = []
        self.tn = cm[0]
        self.fp = cm[1]
        self.fn = PositivesDGA2(cm[2])
        self.tp = PositivesDGA2(cm[3])


def vectorize_cms_3DGA(cms: CMDGA3Type) -> CMVectorizedDGA3Type:
    """
    Return NTH confusion matrix in that format:
    [ TN, FP, [ FNall, FN1, FN2, FN3 ], [ TPall, TP1, TP2, TP3 ] ]
    """

    return (
        np.asarray([ cm[TN] for cm in cms ]),
        np.asarray([ cm[FP] for cm in cms ]),
        tuple([ np.asarray([ cm[FN][dga] for cm in cms ]) for dga in [ 0, 1, 2, 3 ] ]),
        tuple([ np.asarray([ cm[TP][dga] for cm in cms ]) for dga in [ 0, 1, 2, 3 ] ])
    )

def vectorize_cms_2DGA(cms: CMDGA3Type) -> CMVectorizedDGA2Type:
    """
    Return NTH confusion matrix in that format:
    [ TN, FP, [ FNall, FN1, FN2 ], [ TPall, TP1, TP2 ] ]
    """

    cms_abs =  vectorize_cms_3DGA(cms)
    
    return (
        cms_abs[TN],
        cms_abs[FP],
        tuple([ cms_abs[FN][0], cms_abs[FN][1] + cms_abs[FN][2], cms_abs[FN][3] ]),
        tuple([ cms_abs[TP][0], cms_abs[TP][1] + cms_abs[TP][2], cms_abs[TP][3] ])
    )

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
        self.windows_3dga = pd.read_csv(os.path.join(ROOT_DIR, f'./functions_output_old/f/{hash}.csv'), index_col=0)
        self.windows = pd.read_csv(os.path.join(ROOT_DIR, f'./functions_output_2dga/f/{hash}.2dga.csv'), index_col=0)

        self.ths = np.linspace(self.windows.wvalue.min(), self.windows.wvalue.max(), num=configs['nth'], dtype=np.float32)

        _path_pkl = os.path.join(ROOT_DIR, f'./functions_output_old/f/{hash}.pickle')
        with open(_path_pkl, 'rb') as f:
            self.cms_pickle = typing.cast(CMDGA3Type, pickle.load(f))
        cms = vectorize_cms_2DGA(self.cms_pickle)
        self.cms_3dga = ConfusionMatrix(cms)

        _path_pkl = os.path.join(ROOT_DIR, f'./functions_output_2dga/f/{hash}.vectorized.pickle')
        with open(_path_pkl, 'rb') as f:
            cms = typing.cast(CMVectorizedDGA2Type, pickle.load(f))
        self.cms = ConfusionMatrix(cms)

        pass

    def convert_2DGA(self):

        old = self.windows.copy()

        dga = self.windows.dga.where(self.windows.dga != 2, 1)
        dga = dga.where(dga != 3, 2)

        self.windows.dga = self.windows.dga.where(self.windows.dga == dga, dga)

        _path_csv = os.path.join(ROOT_DIR, f'./functions_output_2dga/f/{self.hash}.2dga.csv')

        tmp = pd.concat([ old.dga, self.windows.dga], axis=1, keys=[ 'old', 'new'])
        if not all(tmp[tmp['old'] == 3]['new'] == 2):
            raise Exception('dga 3 not all converted to 2')
        if not all(tmp[tmp['old'] == 2]['new'] == 1):
            raise Exception('dga 2 not all converted to 1')
        tmp_rest = tmp[(tmp['old'] != 2) & (tmp['old'] != 3)]
        if not all(tmp_rest['new'] == tmp_rest['old']):
            raise Exception('some not-to-change value has been changed')

        self.windows.to_csv(_path_csv)

        _path_pkl = os.path.join(ROOT_DIR, f'./functions_output_2dga/f/{self.hash}.vectorized.pickle')

        cms = vectorize_cms_2DGA(self.cms_pickle)
        with open(_path_pkl, 'wb') as f:
            pickle.dump(cms, f)

        pass

    def verify(self):

        if not all(self.cms_3dga.tp.all == self.cms.tp.all):
            raise Exception('\'all\' not the same')

        if not all(self.cms_3dga.tp.dga_1 == self.cms.tp.dga_1):
            raise Exception('\'dga_1\' not the same')

        if not all(self.cms_3dga.tp.dga_2 == self.cms.tp.dga_2):
            raise Exception('\'dga_2\' not the same')


        pass

    @property
    def tnr(self) -> TPVector:
        return self.cms.tn / (self.cms.tn + self.cms.fp)

    @property
    def recall(self) -> PositivesDGA2:
        d = {}
        for dga in ['all', 'dga_1', 'dga_2']:
            d[dga] = self.cms.tp[dga] / (self.cms.tp[dga] + self.cms.fn[dga])
        return PositivesDGA2.from_dict(typing.cast(PositivesDGA2Dict, d))
    
    @property
    def precision(self) -> PositivesDGA2:
        d = {}
        for dga in ['all', 'dga_1', 'dga_2']:
            d[dga] = self.cms.tp[dga] / (self.cms.tp[dga] + self.cms.fp)

        return PositivesDGA2.from_dict(typing.cast(PositivesDGA2Dict, d))
    
    @property
    def accuracy(self) -> PositivesDGA2:
        d = {}
        for dga in ['all', 'dga_1', 'dga_2']:
            d[dga] = self.cms.tp[dga] / ((self.cms.tp[dga] + self.cms.fn[dga]) + self.cms.fp)

        return PositivesDGA2.from_dict(typing.cast(PositivesDGA2Dict, d))

    
    @property
    def f1score(self) -> PositivesDGA2:
        pr = self.precision
        re = self.recall
        
        f1: PrecisionsMetricType = {}  # type: ignore
        for a in [ 'all', 'dga_1', 'dga_2']:
            f1[a] = (2 * pr[a] * re[a]) / (pr[a] + re[a])
        
        return PositivesDGA2.from_dict(typing.cast(PositivesDGA2Dict, f1))