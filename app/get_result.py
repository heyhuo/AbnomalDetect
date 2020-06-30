import datetime
import random
import numpy as np
from flask import json, request

from app.home import home
from app.lib.model import *
from app.lib.options import *

# 生成唯一的图片的名称字符串，防止图片显示时的重名问题
from app.lib.utils import save_gen_images, save_images, get_image


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def create_uuid():
    nowTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S");  # 生成当前时间
    randomNum = random.randint(0, 100);  # 生成的随机整数n，其中0<=n<=100
    if randomNum <= 10:
        randomNum = str(0) + str(randomNum);
    uniqueNum = str(nowTime) + str(randomNum);
    return uniqueNum


def get_result_img(dir_path, path, mean_lamda):
    cwd_path = os.getcwd()
    # print("cwd__path")
    # print(cwd_path)
    # print("dir_path")
    # print(dir_path)
    if cwd_path != dir_path:
        os.chdir(dir_path)
    path = dir_path + path
    if path[-1] == '/':
        path = path[:len(path) - 1]
    sess = tf.Session()
    opt = Options().parse()

    img = get_image(path, 256, 256, 256, 256, False)
    h, w = 32, 32
    batch_x = []
    for i in range(8):
        for j in range(8):
            batch_x.append(img[i * w:i * w + w, j * h:j * h + h, :])
    batch_x = np.array(batch_x).astype(np.float32)

    img_shape = [opt.batchsize, opt.isize, opt.isize, 3]
    img_input = tf.placeholder(tf.float32, img_shape)
    is_train = tf.placeholder(tf.bool)
    labels_out, scores_out = [], []

    with tf.Session() as sess:
        img_gen, latent_z, latent_z_gen = Net_Generator(img_input, opt, istrain=False)

        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('checkpoint'))
        gen_img, latent_i, latent_o, = sess.run([img_gen, latent_z, latent_z_gen],
                                                {img_input: batch_x, is_train: False})

        i_error = np.mean(latent_i, axis=-1)
        i_error = np.reshape(i_error, [-1])
        # i_error = (i_error - i_error.min()) / (i_error.max() - i_error.min())


        o_error = np.mean(latent_o, axis=-1)
        o_error = np.reshape(o_error, [-1])
        # o_error = (o_error - o_error.min()) / (o_error.max() - o_error.min())

        latent_error = np.mean(abs(latent_i - latent_o), axis=-1)
        # print(latent_error)

        # print(latent_error)

        latent_error = np.reshape(latent_error, [-1])
        # latent_error = (latent_error - latent_error.min()) / (latent_error.max() - latent_error.min())

        labels, errors, index, total, normal, abnormal, i_vec, o_vec,ablabels = [], [], [], [], [], [], [], [],[]

        i_latent, o_latent,ablabels = get_latent(latent_i, latent_o,59)

        mean_error = np.mean(latent_error)
        std_error = np.std(latent_error)

        # print(mean_lamda)
        std_error = std_error * 100
        Q1 = np.percentile(latent_error, 25)
        Q3 = np.percentile(latent_error, 75)
        IQR = Q3 - Q1

        th = (Q3 + np.math.log10(std_error*std_error) * IQR)*100
        print(th)
        if std_error < 3:
            thread = (mean_error) * 100 + (mean_lamda * (20 / std_error))
        else:
            thread = (mean_error) * 100 + (mean_lamda * (np.math.log(std_error*std_error)+1))

        print(thread)
        for i, s in enumerate(latent_error):
            # print(i, round(s*100, 4), round(abs(i_error[i]) * 100, 4), round(abs(o_error[i]) * 100, 4),
            #       round(abs(i_error[i] - o_error[i]) * 100, 4))
            # 判断为反例
            index.append(i + 1)
            total.append(round(s * 100, 2))
            i_vec.append(round(abs(i_error[i]) * 100, 4))
            o_vec.append(round(abs(o_error[i]) * 100, 4))
            if s * 100 >= thread:

                abnormal.append([i + 1, round(float(s*100), 4)])
                labels.append(0)
            else:
                normal.append([i + 1, round(float(s*100), 4)])
                labels.append(1)
        # print(mean_error)
        # print(std_error)
        # print(abnormal)
        # print(normal)
        uuniname = create_uuid()
        gen_path_labels = 'static/result_img/' + uuniname + '_feakes_labels.png'
        gen_path = 'static/result_img/' + uuniname + '_feakes.png'
        real_path = 'static/result_img/' + uuniname + '_reals.png'
        save_images(gen_img, [8, 8], gen_path)
        save_gen_images(gen_img, [8, 8], gen_path_labels, labels)
        save_images(batch_x, [8, 8], real_path)
        # print(total)


        return real_path, gen_path, gen_path_labels, index,ablabels, json.dumps(latent_error, cls=NpEncoder), json.dumps(
            mean_error, cls=NpEncoder), json.dumps(std_error, cls=NpEncoder), json.dumps(thread, cls=NpEncoder),total, normal, abnormal,i_vec,o_vec,i_latent,o_latent


def get_latent(latent_i, latent_o,latent_index):
    i_latent, o_latent = [], []
    latent_i = np.reshape(latent_i, [64, 100])
    a = np.reshape(latent_i[latent_index], [-1])
    b = np.reshape(latent_o[latent_index], [-1])
    a.sort()
    b.sort()
    a_mean = np.mean(a)
    a_index = 0
    for i in range(100):
        if a[i] > a_mean:
            a_index = i
            break
    print(a_index)
    a = np.append(a[:a_index], sorted(a[a_index:], reverse=True))
    b = np.append(b[:a_index], sorted(b[a_index:], reverse=True))
    ablabels = []
    for i in range(100):
        i_latent.append(round(float(a[i])*10, 4))
        o_latent.append(round(float(b[i])*10, 4))
        ablabels.append(i)
    print(i_latent)
    return i_latent, o_latent,ablabels
