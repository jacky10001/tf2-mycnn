# -*- coding: utf-8 -*-
"""
基於 tf.keras.Model 的實例物件，實現功能擴展
"""

import os
import os.path as osp
from contextlib import redirect_stdout

import numpy as np
from tensorflow.keras import callbacks, models

from ._file import save_json
from ._history import (find_all_ckpts, find_best_ckpt, find_last_ckpt,
                       load_history, show_history)
from ._wrapper import check_filepath, check_state


class implement_model:
    def __init__(self, method):
        self.method = method
    
    def __get__(self, instance, owner):
        def wrapper(*args, **kwargs):
            target = owner() if instance == None else instance
            mth_name = self.method.__name__

            if mth_name == "setup_model":
                name = kwargs.get('name', "cnn")
                inputs, outputs = self.method(owner, *args, **kwargs)
                target._KerasModel__M = models.Model(inputs=inputs, outputs=outputs, name=name)
            
            if mth_name == "load_from_json":
                json_data = open(self.method(owner, *args, **kwargs), "r").read()
                target._KerasModel__M = models.model_from_json(json_data)
            
            if mth_name == "load_model":
                filepath = self.method(owner, *args, **kwargs)
                target._KerasModel__M = models.load_model(filepath)
            
            weights_path = kwargs.get('weights_path', "")
            if osp.exists(weights_path):
                print('[Info] Pre-trained weights:',  weights_path)
                target._KerasModel__M.load_weights(weights_path)
            
            if kwargs.get('verbose', False):
                target._KerasModel__M.summary()
            
            target.built = True
            target.training = False
            return target
        return wrapper


class KerasModel(object):
    """
    KerasModel
    基於 tf.keras.Model 的實例物件，實現功能擴展
    
    Feature:
    - 覆寫 `build()` 方法來配置 Keras 模型
    - 呼叫 `setup_training()` 方法來定義訓練參數
    - 呼叫 `show_history()` 方法來繪製訓練過程
    - 允許存取 tf.keras.Model 原來方法
    """

    def __init__(self, prebuilt=True, **kwargs) -> None:
        self.__M = models.Model
        self.__cbks_list = []
        self.logdir = kwargs.get('logdir', "log")
        self.built = False
        self.training = False
        if kwargs.get('by_json', ""):
            self.load_from_json(kwargs['by_json'])
        elif kwargs.get('by_h5df', ""):
            self.load_model(kwargs['by_h5df'])
        elif prebuilt:
            self.build()
        else:
            print("[Warning] Please remember to "
                  "first calling `build()` for setup model.")
    
    def build(self, *args, **kwargs) -> None:
        """
        建立神經網路模型方法
        1. 覆寫 method 的方式，來搭建神經網路架構
        2. 允許自行定義所需參數，一定要加上`**kwargs`
        3. 呼叫 self.setup_model(inputs, outputs)

        e.g.
        class MyCNN(KerasModel):
            def build(self, shape=(20,), **kwargs):
                x_in = layers.Input(shape=shape)
                x = layers.Dense(1)(x_in)
                output = layers.Activation("sigmoid")(x)
                self.setup_model(x_in, x_out)
        """
        raise NotImplementedError('[Error] Unimplemented `build()`: '
                                  'Please overridden `build()` '
                                  'method and provide arguments.')

    @implement_model
    def setup_model(self, inputs, outputs, *args, **kwargs) -> tuple:
        """ 載入 JSON 結構檔，來建立神經網路模型 """
        return inputs, outputs
    
    @implement_model
    def load_from_json(self, filepath: str, **kwargs) -> str:
        """ 載入 JSON 結構檔，來建立神經網路模型 """
        if not osp.exists(filepath):
            raise FileNotFoundError(
                f"[Error] No such json file: {filepath} "
                "Please check `filepath` is correct?")
        return filepath
    
    @implement_model
    def load_model(self, filepath: str, **kwargs) -> str:
        """ 載入 H5DF 檔，來載入完整神經網路模型 """
        if not osp.exists(filepath):
            raise FileNotFoundError(
                f"[Error] No such model file: {filepath} "
                "Please check `filepath` is correct?")
        return filepath
    
    @check_state("built")
    def summary(self) -> None:
        self.__M.summary()
    
    def setup_logfile(self, logdir: str):
        self.logdir = logdir
        self.ckpts_dir = osp.join(self.logdir, "weights")
        self.csv_filepath = osp.join(self.logdir, "history.csv")
        if not osp.exists(self.ckpts_dir):
            print('[Info] Create new the directory for training log !!!!!')
            os.makedirs(self.ckpts_dir, exist_ok=True)
        return self

    @check_state("built", "training")
    def add_callback(self, cbk):
        keras_cnks_list = list(filter(lambda x: isinstance(x, type), callbacks.__dict__.values()))
        if type(cbk) not in keras_cnks_list:
            raise Exception("[Error] Please check your callbacks is `keras.callbacks`.")
        self.__cbks_list.append(cbk)

    @check_state("built", "training")
    def add_callbacks(self, cbks):
        if not isinstance(cbks, list):
            raise Exception("[Error] Please check your callbacks is `list`.")
        self.__cbks_list += cbks

    @check_state("built")
    def setup_training(self,
                       logdir,
                       epochs,
                       batch_size,
                       optimizer,
                       loss,
                       metrics=[],
                       best_val=np.inf,
                       **kwargs):        
        self.setup_logfile(logdir)

        ckpts_filename = "weights.{epoch:05d}-{val_loss:.7f}.h5"
        all_ckpts = callbacks.ModelCheckpoint(osp.join(self.ckpts_dir, ckpts_filename), save_weights_only=True)
        all_ckpts.best = best_val

        self.best_filepath = osp.join(self.logdir, "weights.h5")
        best_ckpts = callbacks.ModelCheckpoint(self.best_filepath, save_weights_only=True, save_best_only=True, verbose=1)
        best_ckpts.best = best_val

        log_csv = callbacks.CSVLogger(self.csv_filepath, separator=',', append=True)
        self.__cbks_list = [all_ckpts, best_ckpts, log_csv]
        if kwargs.get('add_callbacks', []):
            self.__cbks_list += kwargs["add_callbacks"]

        initial_epoch, last_ckpt_path = find_last_ckpt(self.ckpts_dir, self.csv_filepath)
        if initial_epoch > 0:
            self.__M.load_weights(last_ckpt_path)

        self.__M.compile(loss=loss, metrics=metrics, optimizer=optimizer)

        self.__cfg = {}
        self.__cfg['checkpoint_path'] = self.ckpts_dir
        self.__cfg['best_checkpoint'] = self.best_filepath
        self.__cfg['epochs'] = epochs
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

    @check_state("built", "training")
    def train(self, tra_x, tra_y, val_x, val_y, last_checkpoint=""):
        if osp.exists(last_checkpoint):
            print(f"[Info] Loading pre-weights from {last_checkpoint}")
            self.__M.load_weights(last_checkpoint)
        elif not osp.exists(last_checkpoint) and last_checkpoint != "":
            raise FileNotFoundError(f"[Error] No such weights file: {last_checkpoint} "
                                    "Please first by calling `setup_logfile()`")
        self.__M.fit(tra_x, tra_y,
                     validation_data=(val_x, val_y), 
                     batch_size=self.__cfg['batch_size'],
                     epochs=self.__cfg['epochs'],
                     initial_epoch=self.__cfg['initial_epoch'],
                     callbacks=self.__cbks_list)

    @check_state("built", "training")
    def train_generator(self, tra_generator, val_generator, last_checkpoint=""):
        if osp.exists(last_checkpoint):
            print(f"[Info] Loading pre-weights from {last_checkpoint}")
            self.__M.load_weights(last_checkpoint)
        elif not osp.exists(last_checkpoint) and last_checkpoint != "":
            raise FileNotFoundError(f"[Error] No such weights file: {last_checkpoint} "
                                    "Please first by calling `setup_logfile()`")
        self.__M.fit(tra_generator, validation_data=val_generator,
                     epochs=self.__cfg['epochs'],
                     initial_epoch=self.__cfg['initial_epoch'],
                     callbacks=self.__cbks_list)

    @check_state("built")
    def predict(self, x):
        return self.__M.predict(x)

    @check_state("built")
    def evaluate(self, x, y):
        return self.__M.evaluate(x=x, y=y, batch_size=self.__cfg['batch_size'])
    
    @check_filepath(mode="save", ext=".json")
    def export_json(self, filepath: str):
        save_json(filepath, self.__M.to_json())
        
    @check_filepath(mode="save", exts=[".h5", ".h5df"])
    @check_state("built")
    def save_weights(self, filepath: str):
        print(f"Save weights to {filepath}")
        self.__M.save_weights(filepath)
        
    @check_filepath(mode="load", exts=[".h5", ".h5df"])
    @check_state("built")
    def load_weights(self, filepath: str):
        self.__M.load_weights(filepath)
        return self
        
    @check_state("built")
    def get_layers_weights(self, verbose=True) -> dict:
        layers_weights = {}
        for ind, layer in enumerate(self.__M.layers):
            if layer.get_weights():
                if verbose:
                    print("[Info]", ind, layer.name)
                layers_weights[ind]= layer.get_weights()
        return layers_weights
        
    @check_state("built")
    def set_layers_weights(self, layers_weights):
        for ind, layer in enumerate(self.__M.layers):
            if ind in layers_weights.keys():
                layer.set_weights(layers_weights[ind])
                print(f"[Info] Set ID-{ind} layer weights")

    @check_state("built")
    def __getattr__(self, name: str) -> any:
        """
        此方法允許存取 self.__M 的所有屬性/方法 (attribute/method)
        因此，所有 tf.models.Model 的方法及可以用 KerasModel 呼叫

        Note: __getattr__() 用法
        只有當呼叫不存在於 KerasModel 的屬性/方法時，才會呼叫此方法
        """
        if name.find("_") != 0:
            print(f"\nUse `tf.keras.Model` method {name}\n")
        return getattr(self.__M, name)

    def show_history(self, metrics: list=['loss'], start=0, end=None):
        if not osp.exists(self.csv_filepath):
            raise FileNotFoundError(f"[Error] No such CSV file: {self.csv_filepath} "
                                    "Please first by calling `setup_logfile()`")
        hist_csv = load_history(self.csv_filepath)
        for metric in metrics:
            show_history(hist_csv[start:end], self.logdir, metric, "val_"+metric)

    def find_all_checkpoints(self):
        if not osp.exists(self.ckpts_dir):
            raise NotADirectoryError(f"[Error] No such directory: {self.ckpts_dir} "
                                     "Please first by calling `setup_logfile()`")
        return find_all_ckpts(self.ckpts_dir)

    def load_best_checkpoint(self):
        if not osp.exists(self.ckpts_dir):
            raise NotADirectoryError(f"[Error] No such directory: {self.ckpts_dir} "
                                     "Please first by calling `setup_logfile()`")
        ckpts = find_all_ckpts(self.ckpts_dir)
        find_best_ckpt(ckpts)
        return self

    @property
    def model(self) -> models.Model:
        return self.__M
    
    @property
    def config(self) -> dict:
        return self.__cfg
    
    @property
    def callbacks(self) -> None:
        for cbk in self.__cbks_list:
            print("[Info]", cbk)
