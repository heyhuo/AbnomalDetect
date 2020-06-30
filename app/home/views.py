# coding:utf8
import datetime
import random
import os
from flask import render_template, request, json, jsonify
from werkzeug.utils import secure_filename
import cv2

from app.get_result import get_result_img
from app.home import home

# from app.get_result import get_result_img
from app.lib.utils import get_image

dir_path = "/Users/huoshan/PycharmProjects/AbnomalDetect/app"
os.chdir(dir_path)


# 生成唯一的图片的名称字符串，防止图片显示时的重名问题
def create_uuid():
    nowTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S");  # 生成当前时间
    randomNum = random.randint(0, 100);  # 生成的随机整数n，其中0<=n<=100
    if randomNum <= 10:
        randomNum = str(0) + str(randomNum);
    uniqueNum = str(nowTime) + str(randomNum);
    return uniqueNum;


@home.route("/")
def index():
    return render_template("home/login.html")


@home.route("/detect")
def detect_img():
    real_img_name = "/static/img/show_real.png"
    fake_img_name = "/static/img/show_fakes.png"
    fake_img_label = "/static/img/fake_label.png"
    return render_template("home/detect.html", real_img_name=real_img_name, fake_img_name=fake_img_name,
                           fake_img_label=fake_img_label)


@home.route("/up_real_img", methods=["POST"])
def up_real_img():
    # print("up_____")
    # print(dir_path)
    mean_lamda = request.values.get('mean_lamda')
    print(mean_lamda)
    img = request.files['real_img']
    fname = secure_filename(img.filename)
    ext = fname.split('.', 1)[1]
    uuniname = create_uuid()
    file_path = os.path.join(dir_path, "static/upload", uuniname + '.' + ext)
    # n_file_path = "static/upload/" + uuniname + '.' + ext
    img.save(file_path)
    fake_img_name = "/static/img/show_fakes.png"
    fake_img_label = "/static/img/fake_label.png"
    file_path = "/static/upload/" + uuniname + '.' + ext

    return render_template("home/detect.html", real_img_name=file_path, fake_img_name=fake_img_name,
                           fake_img_label=fake_img_label)


@home.route('/detect_img', methods=['POST'])
def show_detect_img():
    data = json.loads(request.form.get('data'))
    img_path = str(data["img_path"])
    mean_lamda = float(data["mean_lamda"])

    real_path, fake_path, fake_label_path, label, ablabels, error, mean, std, thread, total, normal, abnormal, i_vec, o_vec,i_latent,o_latent = get_result_img(
        dir_path, img_path,
        mean_lamda)
    data = jsonify({
        'real_path': real_path,
        'fake_path': fake_path,
        'fake_label_path': fake_label_path,
        'label': label,
        'ablabels': ablabels,
        'error': error,
        'mean': mean,
        'std': std,
        'thread': thread,
        'total': total,
        'normal': normal,
        'abnormal': abnormal,
        'i_vec': i_vec,
        'o_vec': o_vec,
        'i_latent':i_latent,
        'o_latent':o_latent
    })
    # print(total)
    # print(label,fake_path)

    return data
