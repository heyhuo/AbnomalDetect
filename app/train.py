import tensorflow as tf
from app.lib.options import *
from app.lib.utils import save_images
from app.lib.visualizer import *
from tqdm import tqdm

from app.lib.model import *


def train(opt,train_img,test_img,test_label):
    '''
     train model and evaluate on the test data by AUC
    :param opt:
    :param train_img:
    :return:
    '''

    ''' build model '''
    with tf.Session() as sess:
        model = Ganomaly(sess, opt)

        ''' strat training '''
        auc_all = []
        for i in range(opt.niter):
            # 所有的训练损失
            loss_train_all = []
            # 所有的测试损失
            loss_test_all = []
            # 原图损失
            real_losses = []
            # 重建图损失
            fake_losses = []
            # 编码器损失
            encoder_loss = []

            # 将每批的数据随机打乱
            permutated_indexes = np.random.permutation(train_img.shape[0])

            for index in tqdm(range(int(train_img.shape[0] / opt.batchsize))):
                batch_indexs = permutated_indexes[index * opt.batchsize: (index + 1) * opt.batchsize]
                batch_x = train_img[batch_indexs]
                # 开始训练
                loss,adv_loss,con_loss,enc_loss = model.train(batch_x)
                loss_train_all.append(loss)
                real_losses.append(adv_loss)
                fake_losses.append(con_loss)
                encoder_loss.append(enc_loss)

            save_train_path = os.path.join('./output/train_img','epoch_'+str(i))
            if not os.path.exists(save_train_path):
                os.makedirs(save_train_path)

            # 保存训练模型
            model.save('./checkpoint',i)



            save_images(batch_x, [8, 8], save_train_path + '/{:d}_real.png'.format(i))
            gen_img = model.get_gen_img(batch_x)
            save_images(gen_img, [8, 8], save_train_path + '/{:d}_fakes.png'.format(i))

            # 打印
            print("iter_epoch => {:>6d} loss_train_all => :{:.4f} adv_loss => {:.4f} con_loss => {:.4f} enc_loss => {:.4f}".
                format(i + 1, np.mean(loss_train_all), np.mean(adv_loss), np.mean(con_loss), np.mean(enc_loss)))

            # if (i + 1) % 1 == 0:
            # 模型评测
            scores_out, labels_out, auc_out = model.evaluate(test_img, test_label)
            print("iter=>[{:>6d}] : Score=>[{}],Label=>[{}],AUC=>[{}]".format(i + 1, scores_out,labels_out,auc_out))
            auc_all.append(auc_out)

    plt.plot(auc_all)
    plt.xlabel('iteration')
    plt.ylabel('AUC value on test dataset')
    plt.grid(True)
    plt.show()



