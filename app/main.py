import random

import cv2

from app.lib.utils import get_image, mixed_pics
from app.train import *
import scipy.misc

x_train = []

def cut_img(path,index):
    img = get_image(path, 300, 400, 256, 256, False)
    h, w = 32, 32
    # batch_x = []
    new_path = "data/new_p"
    for i in range(8):
        for j in range(8):
            new_img = img[i * w:i * w + w, j * h:j * h + h, :]
            # print(index*64+i+j)
            scipy.misc.imsave(os.path.join(new_path,str(index*64+i*8+j)+".jpg"), new_img)

def get_cut(path):
    cut_path = dataset_files(os.getcwd() + "/data/p")
    for i, path in enumerate(cut_path):
        if i%10==0: print(i)
        cut_img(path, i)
        # img = get_image(path, 32, 32, 32, 32, False)
    print('done.')

def get_test():
    # rust_path = dataset_files(os.getcwd() + "/data/test/new_rust")
    # for i,rust in enumerate(rust_path):
    #     img = cv2.imread(rust)
    #     res = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
    #     cv2.imwrite('data/test/new_0/' + str(i) + '.jpg', res)

    rust_path = dataset_files(os.getcwd() + "/data/test/cut_rust")
    test_path = dataset_files(os.getcwd() + "/data/test/before_0")
    print(test_path)
    for i,img in enumerate(test_path):
        rust_name= rust_path[random.randint(1, len(rust_path))-1]
        # hs,ws=random.randint(5,10)/10,random.randint(5,10)/10
        # mixed_img = mixed_pics(rust_name, img, hs, ws)
        rust_img,cell_img= cv2.imread(rust_name),cv2.imread(img)
        mixed_img = cv2.add(rust_img,cell_img)
        cv2.imwrite('data/test/0/'+str(i)+'.jpg',mixed_img)
        if i%100==0:print(i)
    print('done.')

if __name__ == "__main__":
        '''制作训练数据集'''
        # get_cut("data/p")

        '''制作测试数据集'''
        # get_test()

        ''' Step0 => prepare PV data'''
        opt = Options().parse()
        is_crop = False

        train_path = dataset_files(os.getcwd() + "/data/train/new_p")
        test_0_path = dataset_files(os.getcwd()+"/data/test/0")
        test_1_path = dataset_files(os.getcwd() + "/data/test/1")
        #
        # # 训练数据集
        #
        x_train,x_test,y_test = [],[],[]
        for i,path in enumerate(train_path):
            img = get_image(path,32,32,32,32,is_crop)
            x_train.append(img)
        # 测试数据集
        for i,path in enumerate(test_0_path):
            img = get_image(path, 32, 32, 32, 32, is_crop)
            x_test.append(img)
            y_test.append(0)
        for i,path in enumerate(test_1_path):
            img = get_image(path, 32, 32, 32, 32, is_crop)
            x_test.append(img)
            y_test.append(1)

        x_train = np.array(x_train).astype(np.float32)
        x_test = np.array(x_test).astype(np.float32)
        y_test = np.array(y_test).astype(np.float32)

        # 训练模型
        train(opt,x_train,x_test,y_test)






