# -*- coding: utf-8 -*-
"""
基於 tf.keras.Model 擴展功能
"""

import os
import os.path as osp
from contextlib import redirect_stdout

import numpy as np
from tensorflow.keras import callbacks, optimizers, models

from ._file import save_json
from ._history import (find_all_ckpts, find_best_ckpt, find_last_ckpt,
                       load_history, show_history)
from ._wrapper import check_filepath, check_state

KERAS_CALLBACKS = list(filter(lambda x: isinstance(x, type), callbacks.__dict__.values()))
KERAS_OPTIMIZERS = list(filter(lambda x: isinstance(x, type), optimizers.__dict__.values()))


class implement_model:
    """
    實例化模型物件 (tf.keras.models API)
    """
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
            self.build(**kwargs)
        else:
            print("[Warning] Please remember to "
                  "first calling `build()` for setup model.")
    
    def build(self) -> None:
        """
        覆寫 method 的方式，來建構神經網路模型
        1. 覆蓋 build() 方法來搭建神經網路架構
        2. 使用 __init__() 方法來自行定義所需參數
        3. 呼叫 self.setup_model(inputs, outputs)
           傳入輸入張量 及輸出層 list

        e.g.
        class MyCNN(KerasModel):
            def __init__(self,
                        input_shape=(20,),
                        classes_num=1,
                        **kwargs):
                self.input_shape = input_shape
                self.classes_num = classes_num
            
            def build(self):
                x_in = layers.Input(shape=self.input_shape)
                x = layers.Dense(self.classes_num)(x_in)
                output = layers.Activation("sigmoid")(x)
                self.setup_model(x_in, x_out)
        """
        raise NotImplementedError('[Error] Unimplemented `build()`: '
                                  'Please overridden `build()` '
                                  'method and provide arguments.')

    @implement_model
    def setup_model(self, inputs, outputs, **kwargs) -> tuple:
        """
        實例化(instance)神經網路模型
        類似於 tf.keras.models.Model 的用法

        Arguments
        inputs:  輸入張量 or 輸入張量列表 (list)，使用 Keras Input API
        outputs: 輸出張量 or 輸出張量列表 (list)，使用 Keras Layer API
        name:    String，神經網路的名稱
        """
        return inputs, outputs
    
    @check_state("built")
    def summary(self) -> None:
        """ 印出模型結構
        複寫原 tf.keras.models.Model API
        """
        self.__M.summary(line_length=120)
    
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
        """ 加入 Keras Callback
        
        Arguments
        cbk: Keras Callback 

        Note
        只允許使用 Keras Callback API，可參考下面網址的總覽 (namespace)
        https://www.tensorflow.org/api_docs/python/tf/keras/callbacks
        """
        if type(cbk) not in KERAS_CALLBACKS:
            raise Exception("[Error] Please check your callbacks is `keras.callbacks`.")
        self.__cbks_list.append(cbk)

    @check_state("built", "training")
    def add_callbacks(self, cbks: list):
        """ 加入 Keras Callback List
        
        Arguments
        cbk: List，項目必須是 Keras Callback 

        Note
        只允許使用 Keras Callback API，可參考下面網址的總覽 (namespace)
        https://www.tensorflow.org/api_docs/python/tf/keras/callbacks
        """
        if not isinstance(cbks, list):
            raise Exception("[Error] Please check your callbacks is `list`.")
        for cbk in cbks:
            self.add_callback(cbk)

    @check_state("built")
    def setup_training(self,
                       logdir: str,
                       epochs: int,
                       batch_size: int,
                       optimizer,
                       loss,
                       metrics=[],
                       best_val=np.inf,
                       **kwargs):
        """ 配置訓練用參數
        設定訓練時所產生檔案的保存路徑
        尋找最佳權重
        自動從終止的地方開始訓練
        啟用訓練功能

        Arguments
        logdir:     紀錄訓練時的各種資料
        epochs:     訓練週期
        batch_size: 批次大小 (僅使用 NumPy Array 時有效)
        optimizer:  訓練權重優化器
        loss:       損失函數
        metrics:    評估函數
        best_val:   最佳 loss 數值 (未來會移除，直接改成自動讀取)

        Keras API 相關
        tf.keras.callbacks
            ModelCheckpoint: 來記錄權重檔
            CSVLogger:       訓練變化過程的紀錄檔
        
        tf.keras.models.Model
            load_weights:    載入權重，這裡用來載入最後終止的權重
            compile:         設定訓練用參數，如優化器、損失函數、評估函數
            summary:         印出模型結構，並存成文字檔並歸檔至 logdir 底下

        tensorflow.keras.utils
            plot_model:      繪製模型結構，並存成圖檔並歸檔至 logdir 底下
        """
        if type(loss) != str and hasattr(loss, '__call__'):
            if loss.__module__.find("tensorflow.python.keras.losses") < 0 \
                and loss.__module__.find("mycnn.losses") < 0:
                raise Exception("[Error] Please check your loss is `keras.losses`.")
        if type(optimizer) != str and hasattr(loss, '__call__'):
            if type(optimizer) not in KERAS_OPTIMIZERS:
                raise Exception("[Error] Please check your optimizer is `keras.optimizers`.")
        
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
        self.__cfg['optimizer'] = optimizer.get_config() if type(optimizer) in KERAS_OPTIMIZERS else optimizer
        self.__cfg['loss'] = loss.__name__ if hasattr(loss, '__call__') else loss
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
        """
        使用 tf.keras.Model 原 fit 方法 (搭配 NumPy Array)

        Arguments
        tra_x, tra_y, val_x, val_y 使用 NumPy Array 進行模型訓練
        last_checkpoint: 載入權重路徑，直接替換掉當前權重 (例如直接設定先前最佳權重或是進行轉移學習)

        Keras API 相關        
        tf.keras.models.Model
            load_weights:    載入權重，這裡用來載入指定路徑的權重
            fit:             傳遞 setup_training 所設定的參數並開始模型訓練
        """
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
    def train_dataset(self, tra_dataset, val_dataset, last_checkpoint=""):
        """
        使用 tf.keras.Model 原 fit 方法 (搭配 tf.data.Dataset)

        Arguments
        tra_dataset, val_dataset 使用 tf.data.Dataset 物件進行模型訓練
        last_checkpoint: 載入權重路徑，直接替換掉當前權重 (例如直接設定先前最佳權重或是進行轉移學習)

        Keras API 相關        
        tf.keras.models.Model
            load_weights:    載入權重，這裡用來載入指定路徑的權重
            fit:             傳遞 setup_training 所設定的參數並開始模型訓練
        """
        if osp.exists(last_checkpoint):
            print(f"[Info] Loading pre-weights from {last_checkpoint}")
            self.__M.load_weights(last_checkpoint)
        elif not osp.exists(last_checkpoint) and last_checkpoint != "":
            raise FileNotFoundError(f"[Error] No such weights file: {last_checkpoint} "
                                    "Please first by calling `setup_logfile()`")
        self.__M.fit(tra_dataset, validation_data=val_dataset,
                     epochs=self.__cfg['epochs'],
                     initial_epoch=self.__cfg['initial_epoch'],
                     callbacks=self.__cbks_list)

    @check_state("built")
    def pred(self, x):
        """
        使用 tf.keras.Model 原 predict 方法
        """
        return self.__M.predict(x, batch_size=self.__cfg['batch_size'])

    @check_state("built")
    def pred_dataset(self, pred_dataset):
        """
        使用 tf.keras.Model 原 predict 方法
        """
        return self.__M.predict(pred_dataset)

    @check_state("built")
    def eval(self, x, y):
        """
        使用 tf.keras.Model 原 evaluate 方法
        """
        return self.__M.evaluate(x=x, y=y, batch_size=self.__cfg['batch_size'])

    @check_state("built")
    def eval_dataset(self, eval_dataset):
        """
        使用 tf.keras.Model 原 evaluate 方法
        """
        return self.__M.evaluate(eval_dataset)
    
    @check_filepath(mode="save", ext=".json")
    def export_json(self, filepath: str):
        """
        改善模型 JSON 結構檔可讀性
        """
        save_json(filepath, self.__M.to_json())
        
    @check_filepath(mode="save", exts=[".h5", ".h5df"])
    @check_state("built")
    def save_weights(self, filepath: str):
        print(f"[Info] Save weights to {filepath}")
        self.__M.save_weights(filepath)
        
    @check_filepath(mode="load", exts=[".h5", ".h5df"])
    @check_state("built")
    def load_weights(self, filepath: str, by_name: bool=False):
        print(f"[Info] Load weights from {filepath}")
        self.__M.load_weights(filepath, by_name=by_name)
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
            print(f"\n[Info] Use `tf.keras.Model` method {name}\n")
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
        best_ckpt = find_best_ckpt(ckpts)
        filepath = osp.join(self.ckpts_dir, best_ckpt)
        self.__M.load_weights(filepath)
        return self

    def load_checkpoint(self, epoch: int=-1):
        if not osp.exists(self.ckpts_dir):
            raise NotADirectoryError(f"[Error] No such directory: {self.ckpts_dir} "
                                     "Please first by calling `setup_logfile()`")
        ckpt = find_all_ckpts(self.ckpts_dir)[epoch]
        filepath = osp.join(self.ckpts_dir, ckpt)
        print(f"Choose {epoch}: {filepath}")
        self.__M.load_weights(filepath)
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
