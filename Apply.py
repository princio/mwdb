
from calendar import c
from dataclasses import dataclass
import hashlib
import json
import os
import pickle
from typing import Any, Generic, NamedTuple, Tuple, TypeVar, Union
import typing
import numpy as np
import numpy.typing as npt
from typing import TypedDict

import pandas as pd
from pyparsing import col

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

class Value2DGA:
    def __init__(self, values: tuple[TPVector, TPVector, TPVector]):
        self.values = values
        pass

    def __getitem__(self, key):
        if type(key) == str:
            index = { 'all': 0, 'dga_1': 1, 'dga_2': 2}[typing.cast(str, key)]
        else:
            index = typing.cast(int, key)
        return self.values[index]
    
    @classmethod
    def init2(cls, all: TPVector, dga_1: TPVector, dga_2: TPVector):
        return cls((all, dga_1, dga_2))

    def __str__(self):
        if len(self.values[0]) == 1:
            return f'( {self.values[0][0]}, {self.values[1][0]}, {self.values[2][0]} )'
        else:
            return f'( {self.values[0]}, {self.values[1]}, {self.values[2]} )'

    pass


class Metric:
    def __init__(self, values: tuple[MetricVector, MetricVector, MetricVector], round: int = 3):
        self.values = values
        self.round = round
        pass

    def __getitem__(self, key):
        if type(key) == str:
            index = { 'all': 0, 'dga_1': 1, 'dga_2': 2}[typing.cast(str, key)]
        else:
            index = typing.cast(int, key)
        return self.values[index]
    
    @classmethod
    def init2(cls, all: MetricVector, dga_1: MetricVector, dga_2: MetricVector):
        return cls((all, dga_1, dga_2))
    
    def __str__(self):
        if len(self.values[0]) == 1:
            v = [ self.values[i][0] if self.round < 0 else round(self.values[i][0], self.round) for i in range(3) ]
            return f'( {v[0]}, {v[1]}, {v[2]} )'
        else:
            return f'( {self.values[0]}, {self.values[1]}, {self.values[2]} )'

    pass
    

class ConfusionMatrix:
    def __init__(self, cm: tuple[ TPVector, TPVector, Value2DGA, Value2DGA]):
        self.tn: TPVector = cm[0]
        self.fp: TPVector = cm[1]
        self.fn: Value2DGA = cm[2]
        self.tp = cm[3]
        self.n = self.tn + self.fp
        self.p = Value2DGA(tuple(
            typing.cast(Value2DGA, self.fn)[dga] + typing.cast(Value2DGA, self.tp)[dga] for dga in [ 0, 1, 2 ]
        ))
        pass

    def __getitem__(self, index: int):
        return ConfusionMatrix((
            np.asarray([ self.tn[index] ], dtype=np.int_),
            np.asarray([ self.fp[index] ], dtype=np.int_),
            Value2DGA(tuple( np.asarray([ self.fn[dga][index] ], dtype=np.int_) for dga in range(3) )),
            Value2DGA(tuple( np.asarray([ self.tp[dga][index] ], dtype=np.int_) for dga in range(3) )),
        ))

    @classmethod
    def from_pickle(cls, _pickle: Any):
        tn = typing.cast(TPVector, _pickle[0])
        fp = typing.cast(TPVector, _pickle[1])
        fn = Value2DGA(_pickle[2])
        tp = Value2DGA(_pickle[3])
        return cls((tn, fp, fn, tp))


    def __str__(self):
        if len(self.tn) == 1:
            return f'{self.tn[0]} {self.fp[0]} {self.fn} {self.tp}'
        else:
            return f'{self.tn} {self.fp} {self.fn} {self.tp}'


    @property
    def tnr(self) -> MetricDatum:
        return self.tn / (self.tn + self.fp)

    @property
    def recall(self) -> Metric:
        v = []
        for dga in [ 0, 1, 2 ]:
            v.append(self.tp[dga] / (self.tp[dga] + self.fn[dga]))
        return Metric(tuple(v))
    
    @property
    def precision(self) -> Metric:
        v = [ ]
        for dga in [ 0, 1, 2 ]:
            v.append(self.tp[dga] / (self.tp[dga] + self.fp))
        return Metric(tuple(v))
    
    @property
    def accuracy(self) -> Metric:
        v = [ ]
        for dga in [ 0, 1, 2 ]:
            den = ((self.tp[dga] + self.fn[dga]) + self.fp)
            v.append(self.tp[dga] / den)
        return Metric(tuple(v))

    @property
    def f1score(self) -> Metric:
        pr = self.precision
        re = self.recall
        
        v = [ ]
        for dga in [ 0, 1, 2 ]:
            den = (pr[dga] + re[dga])
            v.append((2 * pr[dga] * re[dga]) / den)
        
        return Metric(tuple(v))
    
    @property
    def plr(self) -> Metric:
        """
        Positive likelihood ratio (LR+)
            = TPR/FPR
        """
        tpr = self.recall
        fpr = self.fp / self.n
        
        v = [ ]
        for dga in [ 0, 1, 2 ]:
            v.append(tpr[dga] / fpr)
        
        return Metric(tuple(v))
    
    @property
    def nlr(self) -> Metric:
        """
        Negative likelihood ratio (LR+)
            = FNR / TNR
            
            where FNR = FN / P = 1 − [TPR|recall]
        """
        re = self.recall
        v = [ ]
        for dga in [ 0, 1, 2 ]:
            v.append(1 - re[dga])
        return Metric(tuple(v))
    
    @property
    def ba(self) -> Metric:
        """
        Balanced accuracy (BA) = 
            =TPR + TNR / 2
        """
        tpr = self.recall
        v = [ ]
        for dga in [ 0, 1, 2 ]:
            v.append((self.tnr + tpr[dga]) / 2)
        return Metric(tuple(v))

    @property
    def ppv(self) -> Metric:
        """
        Positive predictive value
        """
        return self.precision
    
    @property
    def tpr(self) -> Metric:
        """
        True positive rate (TPR), recall, sensitivity (SEN), probability of detection, hit rate, power
        """
        return self.recall
    
    @property
    def npv(self) -> Metric:
        """
        Negative predictive value
        """
        v = [ ]
        for dga in [ 0, 1, 2 ]:
            v.append(self.tn / (self.fn[dga] + self.tn))
        return Metric(tuple(v))
    
    @property
    def fnr(self) -> Metric:
        """
        Negative predictive value
        """
        v = [ ]
        for dga in [ 0, 1, 2 ]:
            v.append(1 - self.tpr[dga])
        return Metric(tuple(v))
    
    @property
    def for_(self) -> Metric:
        """
        False omission rate (FOR)
        = FN / PN = 1 - NPV
        """
        v = [ ]
        for dga in [ 0, 1, 2 ]:
            v.append(1 - self.npv[dga])
        return Metric(tuple(v))
    
    @property
    def fdr(self) -> Metric:
        """
        False discovery rate (FDR)
        = FP / PP  = 1 - PPV
        """
        v = [ ]
        for dga in [ 0, 1, 2 ]:
            v.append(1 - self.ppv[dga])
        return Metric(tuple(v))
    
    @property
    def fpr(self) -> MetricDatum:
        """
        False positive rate (FPR), probability of false alarm, fall-out
        = FP / N = 1 - TNR
        """
        return 1 - self.tnr

    @property
    def phi(self):
        """
        Balanced accuracy (BA) = 
            A = ( TP x TN ) - ( FP x FN )
            B = ( TP + FP ) x ( TP + FN ) x ( TN + FP ) ( TN + FN )
            phi = A / sqrt(B)
        """
        v = [ ]
        for dga in [ 0, 1, 2 ]:
            A =     self.tp[dga] * self.tn
            A = A - self.fp * self.fn[dga]
            B =  ( self.tp[dga] + self.fp)
            B *= ( self.tp[dga] + self.fn[dga])
            B *= ( self.tn + self.fp)
            B *= ( self.tn + self.fn[dga])
            v.append(A / np.sqrt(B))   # type: ignore
        return Metric(tuple(v))

    def metric(self, metric_name) -> Metric:
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
    """
    A class used to represent an Animal

    ...

    Attributes
    ----------
    hash : str
        the hash string of the apply configuration
    windows : str
        the DataFrame of the windows for the apply.
        [ index, wlen, wvalue, wnum, dga, pcap_id ]
    Methods
    -------
    """
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
            self.cm = ConfusionMatrix.from_pickle(pickle.load(f))

        pass

    def pcaps(self, th):
        wnum = self.windows.groupby(['pcap_id', 'dga']).count().wlen.rename('wnum')
        
        legits = self.windows[self.windows.dga == 0]
        tn = (legits.wvalue <= th).sum()
        fp = legits.shape[0] - tn

        infected = self.windows[self.windows.dga > 0].copy()
        infected['tp'] = infected.wvalue > th
        p = infected.groupby(['pcap_id', 'dga']).count().wlen  # type: ignore
        tp = infected.groupby(['pcap_id', 'dga']).sum()['tp']  # type: ignore
        fn = p - tp

        fp = pd.Series(np.repeat(fp, tp.shape[0]), index=tp.index)
        tn = pd.Series(np.repeat(tn, tp.shape[0]), index=tp.index)

        # phi_coeff = tp * tn - fp * fn
        # phi_coeff = phi_coeff / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        result = pd.concat([
            tn,
            fp,
            fn,
            tp,
            tp - fp,
            (tp - fp) > 0,
            # phi_coeff
        ],
        keys=[ 'tn', 'fp', 'fn', 'tp', 'tp - fp', 'detected' ],
        axis=1)

        detected = result.detected.reset_index().groupby('dga').sum()
        pcaps_num_by_dga = result.detected.reset_index().groupby('dga').count()
        undetected = detected.rsub(pcaps_num_by_dga).rename(columns={'detected': 'undetected'})
        
        result = pd.concat([
            undetected.drop(columns='pcap_id'),
            detected.drop(columns='pcap_id')
        ], axis=1)

        result.loc['all'] = result.sum()
        result['tot'] = result.sum(axis=1)
        result['%'] = result.detected / result.tot # ((result.detected / result.tot) * 100).astype(int)
        result = result.T[['all', 1, 2]]
        
        return result.loc['%'].to_numpy().tolist()

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

        return ConfusionMatrix((
            np.asarray([ tn ], dtype=np.int_),
            np.asarray([ fp ], dtype=np.int_),
            Value2DGA(tuple(np.asarray([ v ], dtype=np.int_) for v in fn)),
            Value2DGA(tuple(np.asarray([ v ], dtype=np.int_) for v in tp)),
        ))