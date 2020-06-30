# coding:utf8
from flask import Blueprint

# 定义蓝图
home = Blueprint("home", __name__)

import app.home.views
