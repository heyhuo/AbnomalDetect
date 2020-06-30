from app.get_result import create_uuid
from app.lib.utils import get_image, mixed_pics
from app.train import *
import scipy.misc
import cv2

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
pos_path = "/data/test/s0"
neg_path = "/data/test/s1"
opt = Options().parse()
colors2 = '#5e4fa2'
colors1 = '#9e0142'
def eval():
    opt = Options().parse()
    is_crop = False

    test_0_path = dataset_files(os.getcwd()+pos_path)
    test_1_path = dataset_files(os.getcwd() + neg_path)


    test,label = [],[]

    for i,path in enumerate(test_0_path):
        img = get_image(path, 32, 32, 32, 32, is_crop)
        test.append(img)
        label.append(0)

    for i,path in enumerate(test_1_path):
        img = get_image(path, 32, 32, 32, 32, is_crop)
        test.append(img)
        label.append(1)

    print("load done.")


    test = np.array(test).astype(np.float32)
    label = np.array(label).astype(np.float32)

    return test,label

def plot_2d(X,Y,name,flag=True):
    fig = plt.figure()
    axes = fig.add_subplot(111)

    for i in range(len(Y)):
        if flag:
            if Y[i] == 0:
                #  第i行数据，及returnMat[i:,0]及矩阵的切片意思是:i：i+1代表第i行数据,0代表第1列数据
                axes.scatter(i, X[i], c=colors1, alpha=0.6, label='fakes')
            if Y[i] == 1:
                axes.scatter(i, X[i], c=colors2, alpha=0.6, label='reals')

        else:
            axes.scatter(i, Y[i],c=colors2,alpha=0.6)


    pdf = PdfPages('data/test/pdf/'+name)
    pdf.savefig()
    # plt.close()
    pdf.close()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Epic Chart')  # 对中文的支持很差！
    plt.show()

def plot_3d(name):
    X = np.loadtxt("data/txt/t199_x_3.txt")
    Y = np.loadtxt("data/txt/t199_y.txt")
    fig = plt.figure()
    # axes = fig.add_subplot(111)
    nc = '#c3553a'
    pc = '#3f7f93'

    fig = plt.figure()
    '''绘制S曲线的3D图像'''
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot_trisurf(X[0:320, 0], X[0:320, 1],X[0:320, 2],rstride=1, cstride=1, cmap='rainbow')
    ax.scatter(X[0:320, 0], X[0:320, 1],X[0:320, 2],c=nc,alpha=0.6, label='fakes')
    ax.scatter(X[320:640,0], X[320:640, 1], X[320:640, 2],c = pc,alpha=0.6,label = 'reals')
    ax.view_init(30, -75)  # 初始化视角

    pdf = PdfPages('data/pdf/' + name+".pdf")
    pdf.savefig()
    # plt.close()
    pdf.close()

    plt.show()


# 原图与生成图之间的误差
def real_fake():
    # X = np.loadtxt("data/test/pdf/big_sco.txt")
    # Y = np.loadtxt("data/test/pdf/big_label.txt")
    # plot_2d(X,Y,"new_latent.pdf")

    test, Y = eval()

    img_shape = [opt.batchsize, opt.isize, opt.isize, 3]
    img_input = tf.placeholder(tf.float32, img_shape)
    is_train = tf.placeholder(tf.bool)
    labels_out, scores_out = [], []

    with tf.Session() as sess:
        img_gen, latent_z, latent_z_gen = Net_Generator(img_input, opt, istrain=False)

        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('checkpoint'))

        index = 1
        X,Z,pro_Z=[],[],[]
        for index in range(int(test.shape[0] / 64)):
            batch_x = test[index * 64:(index + 1) * 64]
            batch_y = Y[index * 64:(index + 1) * 64]
            gen_img, latent_i, latent_o, = sess.run([img_gen, latent_z, latent_z_gen],
                                                {img_input: batch_x, is_train: False})
            # latent_o = latent_o.reshape(64,100)
            # latent_i = latent_i.reshape(64,100)

            # tsne = TSNE(n_components=1, init='pca', random_state=0)
            # latent_i = tsne.fit_transform(latent_i)
            # latent_o = tsne.fit_transform(latent_o)

            # uuniname = create_uuid()
            # gen_path = 'static/result_img/' + uuniname + '_feakes.png'
            # real_path = 'static/result_img/' + uuniname + '_reals.png'
            # save_images(gen_img, [8, 8], gen_path)
            # save_images(batch_x, [8, 8], real_path)

            io_error = np.mean(abs(latent_i - latent_o), axis=-1)
            io_error = np.reshape(io_error, [-1])

            X = np.append(X,io_error*100)


        xname = "x_3.txt"
        yname = "y_3.txt"
        np.savetxt("data/test/pdf/" + xname, X)
        np.savetxt("data/test/pdf/" + yname, Y)


        # X = np.loadtxt("data/test/pdf/n_error.txt")
        # Y = np.loadtxt("data/test/pdf/label_2.txt")
        # plot_2d(X,Y,"error_3.pdf")
        # plot_2d(Z,pro_Z,"std_1.pdf")
        # print(auc_out)

def std_plot():
    n = np.loadtxt("data/test/pdf/n_error.txt")
    p = np.loadtxt("data/test/pdf/p_error.txt")
    # Y = np.loadtxt("data/test/pdf/label.txt")

    x,std = [],[]
    for index in range(1):
        for i in range(64):
            error = []
            n_index = np.random.randint(0,320,i)
            p_index = np.random.randint(0,320,64-i)
            for _,e in enumerate(n_index):
                error.append(n[e])
            for _, e in enumerate(p_index):
                error.append(p[e])
            std = np.append(std,np.std(error))
            x = np.append(x,i)

    x = np.array(x).astype(np.float32)
    std = np.array(std).astype(np.float32)
    # print(std)
    plot_2d(x,std,"std_dis_1.pdf",False)
        # print(p_error)

def save_latent():
    test, Y = eval()
    # e = np.loadtxt("data/test/pdf/x_3.txt")
    # p = np.loadtxt("data/test/pdf/p_error_2.txt")
    # n = np.loadtxt("data/test/pdf/n_error_2.txt")

    img_shape = [opt.batchsize, opt.isize, opt.isize, 3]
    img_input = tf.placeholder(tf.float32, img_shape)
    is_train = tf.placeholder(tf.bool)
    labels_out, scores_out = [], []

    with tf.Session() as sess:
        img_gen, latent_z, latent_z_gen = Net_Generator(img_input, opt, istrain=False)

        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('checkpoint'))

        index = 1
        X, Z, pro_Z = [], [], []
        for index in range(int(test.shape[0] / 64)):
            batch_x = test[index * 64:(index + 1) * 64]
            batch_y = Y[index * 64:(index + 1) * 64]
            gen_img, latent_i, latent_o, = sess.run([img_gen, latent_z, latent_z_gen],
                                                    {img_input: batch_x, is_train: False})

            latent_i = np.reshape(latent_i,(64,100))
            latent_o = np.reshape(latent_o,(64, 100))
            np.savetxt("data/test/latent/"+str(index)+"_i.txt",latent_i)
            np.savetxt("data/test/latent/" + str(index) + "_o.txt", latent_o)

    # print(len(e),len(p),len(n))
    # std = []
    # for index in range(1):
    #     for i in range(64):
    #         x,error = [],[]
    #         n_index = np.random.randint(0, 320, i)
    #         p_index = np.random.randint(0, 320, 64 - i)
    #         for _, e in enumerate(n_index):
    #             x = np.append(x, 0)
    #             error.append(n[e])
    #         for _, e in enumerate(p_index):
    #             error.append(p[e])
    #             x = np.append(x, 1)
    #         # std = np.append(std, np.std(error))
    #         # x = np.append(x, i)
    #         error = np.array(error).astype(np.float32)
    #
    #         plot_2d(error,x, "error_dis_" + str(i) + ".pdf")
    # a = 5*np.log2(20)





def plot_dis(dim=3):

    # 模型评估
    with tf.Session() as sess:
        model = Ganomaly(sess, opt)

        ''' strat training '''
        auc_all = []

        # scores_out, labels_out, auc_out = model.evaluate(x_test, y_test)
        # return scores_out,labels_out,auc_out
        test,label = eval()
        X, Y = model.get_whole_dis(test, label)

        print("t-sne...")
        tsne = TSNE(n_components=dim, init='pca', random_state=0)
        X = tsne.fit_transform(X)

    #
    print("save txt...")
    if dim==2:
        xname = "t199_x_2.txt"
    else : xname = "t199_x_2.txt"
    np.savetxt("data/test/pdf/"+xname, X)
    np.savetxt("data/test/pdf/t199_y.txt", Y)

    # X_2 = np.loadtxt("data/test/pdf/t199_x_2.txt")
    # X_3 = np.loadtxt("data/test/pdf/t199_x_3.txt")
    # Y = np.loadtxt("data/test/pdf/t199_y.txt")
    # name_2 = 't199_2.pdf'
    # name_3 = 't199_3.pdf'
    # plot_2d(X_2, Y, name_2)
    # plot_3d(X_3, Y, name_3)


def judge_fake():
    p_latent = np.loadtxt("data/test/latent/9_i.txt")
    n_latent = np.loadtxt("data/test/latent/9_o.txt")

    p = p_latent[0]
    n = n_latent[0]
    # plt.hist(p)
    plt.hist(p, color=colors1, alpha=.2)

    plt.hist(n , color=colors2, alpha=.2)
    plt.show()
    # print(p)


if __name__=="__main__":
    # plot_dis()
    # real_fake()
    # std_plot()
    # judge_fake()
    plot_3d("aaa")

