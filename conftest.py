"""pytest 根目录配置：将仓库根加入 sys.path 以便 import rankforge。"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
