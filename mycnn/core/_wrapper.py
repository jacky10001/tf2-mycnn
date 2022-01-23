# -*- coding: utf-8 -*-
"""
用 Python 裝飾器包裝 KerasModel 類的方法

- 檢查模型的狀態
- 檢查路徑

Note: 裝飾器可以用來延伸 Python 函式功能
也可以用來把函式傳遞的引數進行額外操作
"""

import os
import os.path as osp


def check_state(*state):
    """
    檢查模型的狀態的裝飾器
    - 是否有成功建立模型
    - 確認是否設定訓練用參數
    """
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
    """
    檢查路徑的裝飾器
    """
    if not isinstance(ext, str):
        raise TypeError("The decorator `ext` must be a string.")
    if not isinstance(exts, list):
        raise TypeError("The decorator `exts` must be a list.")
    def decorator(mth):
        def wrapper(self, *args, **kwargs):
            filepath = None
            if isinstance(pos, int):
                filepath = args[pos]
            if isinstance(pos, str):
                filepath = kwargs.get(pos, None)
            if not isinstance(filepath, str):
                raise TypeError(
                    "[Error] The argument `filepath` must be a string."
                )
            if mode == "load":
                if not osp.exists(filepath):
                    raise FileNotFoundError(
                        "[Error] No such model file: {filepath} "
                        "Please check `filepath` is correct?"
                    )
            if exts != [] or ext != "":
                ext_list = [ext]+exts
                if osp.splitext(filepath)[-1] not in ext_list:
                    raise FileNotFoundError(
                        "[Error] No such model file: {filepath} "
                        "Please check `filepath` extention name is correct?"
                    )
            return mth(self, *args, **kwargs)
        return wrapper
    return decorator

