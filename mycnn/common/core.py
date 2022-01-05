# -*- coding: utf-8 -*-
"""
將 Keras API 進行封裝

- 覆寫 build() 方法來配置 Keras 模型
- 訓練相關參數由 setup_training() 方法來定義
- 一些常用 API 包成類別的方法來呼叫

Note: 藉由實例化此類別，即可重現先前開發的功能
"""

import os
import os.path as osp
import re
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
from numpy.core import overrides
import pandas as pd
from tensorflow.keras import callbacks, models

from ..utils.file import *


def save_history(history, result_path, mode='a'):
    """
    紀錄Keras模型訓練的過程 (紀錄fit的最終結果)
    
    Keras 模型在 fit 執行結束之後，會返回一個 history 的 dict
    此程式目的為將此訓練過程紀錄保存下來，以方便後續分析查看
    """
    if not(os.path.exists(result_path)) or mode != 'a':
        # 檔案不存在 或 覆蓋檔案 時，需要加上標題，
        header=True
    else:
        header=False
    pd.DataFrame(history).to_csv(result_path, encoding='utf-8', mode=mode,
                                 index=False, header=header)


def load_history(result_path):
    """
    讀回Keras模型訓練的過程紀錄檔案
    """
    history = pd.read_csv(result_path)
    return history


def show_history(history, savepath, *keys):
    """
    可視化訓練過程中的loss、accuracy...等指標
    """
    for key in keys:
        plt.plot(history[key])
    plt.title('Train History')
    plt.xlabel('Epoch')
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=2)
    
    if keys[0]:
        plt.savefig(os.path.join(savepath, "history_%s"%keys[0]))
    plt.show()
    

def find_all_ckpts(log_ckpt_path, only_filename=True):
    """
    在指定資料夾中，找尋全部儲存的權重路徑 (*依靠副檔名來判斷)
    配合 Keras 的 callbacks 函數 ModelCheckpoint 使用
    """
    checkpoints = next(os.walk(log_ckpt_path))[2]
    if only_filename:
        return checkpoints
    else:
        ckpts_list = []
        for ckpt in checkpoints:
            ext = os.path.splitext(ckpt)[-1]
            if ext == ".h5" or ext == ".h5df":
                ckpts_list.append(os.path.join(log_ckpt_path,ckpt))
        print("Find %d checkpoint"%len(ckpts_list))
        return ckpts_list
    

def find_last_ckpt(log_ckpt_path, log_history_path):
    """
    在指定資料夾中，找尋最後儲存的權重路徑 (*依靠檔名來判斷訓練的epoch)
    配合 Keras 的 callbacks 函數 ModelCheckpoint 使用
    """
    checkpoints = find_all_ckpts(log_ckpt_path)
    if checkpoints:
        print("*"*30)
        print("Loading last checkpoint...")
        last_ckpt_path = os.path.join(log_ckpt_path,checkpoints[-1])
        init_epoch = len(checkpoints)
        
        ## 更新訓練過程紀錄 (手動刪除權重時用)
        hist_csv = load_history(log_history_path)
        save_history(hist_csv[:init_epoch], log_history_path, mode='w+')
        print("Start traing from epoch %d"%init_epoch)
        print("*"*30)
    else:
        print("*"*30+"\nTraining new model...\n"+"*"*30)
        last_ckpt_path = ""
        init_epoch = 0
    return init_epoch, last_ckpt_path
    

def find_best_ckpt(ckpts, pattern=r"weights.(\d{5})-(\d{1,2}.\d{7})", mode="min"):
    """
    在指定資料夾找尋最佳權重結果 (檔名需配合 pattern)
    配合 Keras 的 callbacks 函數 ModelCheckpoint 使用
    """
    if mode.lower() == "min":
        threshold = np.inf
    elif mode.lower() == "max":
        threshold = -np.inf
    best_idx = 0
    for i in ckpts:
        ckpt = re.match(pattern, i)
        ep, val = ckpt.groups()
        ep = int(ep)
        val = float(val)
        if val < threshold and mode.lower() == "min":
            threshold = val
            best_idx = ep
        elif val > threshold and mode.lower() == "max":
            threshold = val
            best_idx = ep
    best_ckpt = ckpts[best_idx]
    print()
    print("best model : {}".format(best_ckpt))
    
    return best_ckpt


def check_args(*state):
    def decorator(mth):
        def wrapper(self, *args, **kwargs):
            if not self.built and "built" in state:
                raise NotImplementedError('Not define model, please call `build()` method.')
            if not self.training and "training" in state:
                raise NotImplementedError('Not define training parameter, please call `setup_training()` method.')
            return mth(self, *args, **kwargs)
        return wrapper
    return decorator


def check_filepath(mode="save", pos=0, ext="", exts=[]):
    if not isinstance(ext, str):
        raise Exception("The decorator `ext` must be a string.")
    if not isinstance(exts, list):
        raise Exception("The decorator `exts` must be a list.")
    def decorator(mth):
        def wrapper(self, *args, **kwargs):
            filepath = None
            if isinstance(pos, int):
                filepath = args[pos]
            if isinstance(pos, str):
                filepath = kwargs.get(pos, None)
            if not isinstance(filepath, str):
                raise Exception("The argument `filepath` must be a string.")
            if mode == "load":
                if not osp.exists(filepath):
                    raise Exception("Please check `filepath` is correct?")
            if exts != [] or ext != "":
                ext_list = [ext]+exts
                if osp.splitext(filepath)[-1] not in ext_list:
                    raise Exception("Please check `filepath` extention name is correct?")
            return mth(self, *args, **kwargs)
        return wrapper
    return decorator

    
class implement_model:
    def __init__(self, method):
        self.method = method
    
    def __get__(self, instance, owner):
        def wrapper(*args, **kwargs):
            target = owner() if instance == None else instance
            mth_name = self.method.__name__
            if mth_name == "build":
                name = kwargs.get('name', "cnn")
                inputs, outputs = self.method(owner, *args, **kwargs)
                target._KerasModel__M = models.Model(inputs=inputs, outputs=outputs, name=name)

            if mth_name == "setup_model":
                name = kwargs.get('name', "cnn")
                inputs, outputs = self.method(owner, *args, **kwargs)
                target._KerasModel__M = models.Model(inputs=inputs, outputs=outputs, name=name)
            
            if mth_name == "load_from_json":
                json_data = open(self.method(owner, *args, **kwargs), "r").read()
                target._KerasModel__M = models.model_from_json(json_data)
            
            weights_path = kwargs.get('weights_path', "")
            if osp.exists(weights_path):
                print('Pre-trained weights:',  weights_path)
                target._KerasModel__M.load_weights(weights_path)
            
            if kwargs.get('verbose', False):
                target._KerasModel__M.summary()
            
            target.built = True
            target.training = False
            return target
        return wrapper


class KerasModel(object):
    """
    KerasModel: 封裝 Keras Model API
    - 覆寫 build() 方法來配置 Keras 模型
    - 訓練相關參數由 setup_training() 方法來定義
    - 一些常用 API 包成類別的方法來呼叫
    """

    def __init__(self):
        self.__M = None
        self.__cbks_list = []
        self.logdir = "log"
        self.built = False
        self.training = False
    
    def build(self, input_shape, *args, **kwargs):
        """
        建立神經網路模型方法
        1. 覆寫 method 的方式，來搭建神經網路架構
        2. 需自行定義所需參數，一定要加上`**kwargs`
        3. 呼叫 self.setup_model(inputs, outputs)

        e.g.
        @implement_model
        def build(self, shape=(20,), **kwargs):
            x_in = layers.Input(shape=shape)
            x = layers.Dense(1)(x_in)
            output = layers.Activation("sigmoid")(x)
            return x_in, x_out
        """
        raise NotImplementedError('Unimplemented `build()`: '
                                  'Please overridden `build()` '
                                  'method and provide arguments.')

    @implement_model
    def setup_model(self, inputs, outputs, *args, **kwargs):
        return inputs, outputs
    
    @implement_model
    def load_from_json(self, json_path, **kwargs):
        """
        載入JSON結構檔，來建立神經網路模型
        """
        if osp.splitext(json_path)[-1] != ".json"\
        or not osp.exists(json_path)\
        or not isinstance(json_path, str):
            raise Exception("Please check \"json_path\" is correct?")
        return json_path

    def summary(self):
        if not self.built:
            raise NotImplementedError('Not define model, please call `build()` method.')
        self.__M.summary()
    
    def setup_logfile(self, logdir):
        self.logdir = logdir
        self.ckpts_filepath = osp.join(self.logdir, "weights")
        self.csv_filepath = osp.join(self.logdir, "history.csv")
        if not osp.exists(self.ckpts_filepath):
            print('Create new the directory for training log !!!!!')
            os.makedirs(self.ckpts_filepath, exist_ok=True)
        return self

    @check_args("built", "training")
    def add_callback(self, cbk):
        keras_cnks_list = list(filter(lambda x: isinstance(x, type), callbacks.__dict__.values()))
        if type(cbk) not in keras_cnks_list:
            raise Exception("Please check your callbacks is `keras.callbacks`.")
        self.__cbks_list.append(cbk)

    @check_args("built", "training")
    def add_callbacks(self, cbks):
        if not isinstance(cbks, list):
            raise Exception("Please check your callbacks is `list`.")
        self.__cbks_list += cbks

    @check_args("built")
    def setup_training(self,
                       logdir,
                       nb_epoch,
                       batch_size,
                       optimizer,
                       loss,
                       metrics=[],
                       best_val=np.inf,
                       **kwargs):        
        self.setup_logfile(logdir)

        ckpts_filename = "weights.{epoch:05d}-{val_loss:.7f}.h5"
        all_ckpts = callbacks.ModelCheckpoint(osp.join(self.ckpts_filepath, ckpts_filename), save_weights_only=True)
        all_ckpts.best = best_val

        self.best_filepath = osp.join(self.logdir, "weights.h5")
        best_ckpts = callbacks.ModelCheckpoint(self.best_filepath, save_weights_only=True, save_best_only=True, verbose=1)
        best_ckpts.best = best_val

        log_csv = callbacks.CSVLogger(self.csv_filepath, separator=',', append=True)
        self.__cbks_list = [all_ckpts, best_ckpts, log_csv]
        if kwargs.get('add_callbacks', []):
            self.__cbks_list += kwargs["add_callbacks"]

        initial_epoch, last_ckpt_path = find_last_ckpt(self.ckpts_filepath, self.csv_filepath)
        if initial_epoch > 0:
            self.__M.load_weights(last_ckpt_path)

        self.__M.compile(loss=loss, metrics=metrics, optimizer=optimizer)

        self.__cfg = {}
        self.__cfg['checkpoint_path'] = self.ckpts_filepath
        self.__cfg['best_checkpoint'] = self.best_filepath
        self.__cfg['nb_epoch'] = nb_epoch
        self.__cfg['initial_epoch'] = initial_epoch
        self.__cfg['batch_size'] = batch_size
        self.__cfg['optimizer'] = optimizer
        self.__cfg['loss'] = loss
        self.__cfg['metrics'] = metrics
        for item in self.__cfg.items():
            print("{:15} : {}".format(*item))
        
        if kwargs.get('export_summary', False):
            # 保存 Keras 模型的 summary 文字
            with open(osp.join(self.logdir, "summary.txt"), 'w+') as f:
                with redirect_stdout(f):
                    self.__M.summary()
        if kwargs.get('export_plot', False):
            # 繪製 Keras 模型的架構圖
            from tensorflow.keras.utils import plot_model
            plot_model(self.__M, to_file=osp.join(self.logdir, "model.png"), show_shapes=True)
        self.training = True
        return self

    @check_args("built", "training")
    def train(self, tra_x, tra_y, val_x, val_y, last_checkpoint=""):
        if osp.exists(last_checkpoint):
            self.__M.load_weights(last_checkpoint)
        self.__M.fit(tra_x, tra_y,
                     validation_data=(val_x, val_y), 
                     batch_size=self.__cfg['batch_size'],
                     epochs=self.__cfg['nb_epoch'],
                     initial_epoch=self.__cfg['initial_epoch'],
                     callbacks=self.__cbks_list)

    @check_args("built", "training")
    def train_generator(self, tra_generator, val_generator, last_checkpoint=""):
        if osp.exists(last_checkpoint):
            self.__M.load_weights(last_checkpoint)
        self.__M.fit_generator(tra_generator,
                               samples_per_epoch=tra_generator.nb_sample,
                               validation_data=val_generator,
                               nb_val_samples=val_generator.nb_sample,
                               epochs=self.__cfg['nb_epoch'],
                               initial_epoch=self.__cfg['initial_epoch'],
                               callbacks=self.__cbks_list)

    @check_args("built")
    def predict(self, x):
        return self.__M.predict(x)
    
    @check_filepath(mode="save", ext=".json")
    def export_json(self, filepath):
        save_json(filepath, self.__M.to_json())
        
    @check_filepath(mode="save", exts=[".h5", ".h5df"])
    @check_args("built")
    def save_weights(self, filepath):
        self.__M.save_weights(filepath)
        
    @check_filepath(mode="load", exts=[".h5", ".h5df"])
    @check_args("built")
    def load_weights(self, filepath):
        self.__M.load_weights(filepath)
        return self
        
    @check_args("built")
    def get_layers_weights(self, verbose=True):
        layers_weights = {}
        for ind, layer in enumerate(self.__M.layers):
            if layer.get_weights():
                if verbose:
                    print(ind, layer.name)
                layers_weights[ind]= layer.get_weights()
        return layers_weights
        
    @check_args("built")
    def set_layers_weights(self, layers_weights):
        for ind, layer in enumerate(self.__M.layers):
            if ind in layers_weights.keys():
                layer.set_weights(layers_weights[ind])

    @property
    def model(self):
        return self.__M
    
    @property
    def config(self):
        return self.__cfg
    
    @property
    def callbacks(self):
        for cbk in self.__cbks_list:
            print(cbk)
