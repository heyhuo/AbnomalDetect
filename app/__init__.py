# coding:utf8
from flask import Flask, render_template
import os
from app.home import home as home_blueprint

app = Flask(__name__)
app.debug = True


# 定义文件上传保存的路径，在__init__.py文件所在目录创建media文件夹，用于保存上传的文件
app.config['UP_DIR'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'static/upload/')

# # 添加全局404页面
# @app.errorhandler(404)
# def page_not_found(error):
#     return render_template('404.html'), 404


# from app.admin import admin as admin_blueprint

app.register_blueprint(home_blueprint)
# app.register_blueprint(admin_blueprint,url_prefix="/admin")
