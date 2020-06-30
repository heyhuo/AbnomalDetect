import itertools
from glob import glob
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Conv2D, LeakyReLU, Conv2DTranspose, Activation, Flatten, Dense


SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg"]

def dataset_files(root):
    """Returns test_img list of all image files in the given directory"""
    return list(itertools.chain.from_iterable(
        glob(root+"/*.{}".format(ext)) for ext in SUPPORTED_EXTENSIONS))

def roc(labels, scores,saveto = None):
    roc_auc = dict()
    # True/False Positive Rates..
    fpr,tpr,_ = roc_curve(labels,scores)
    roc_auc = auc(fpr,tpr)
    return roc_auc


def l1_loss(y_true,y_pred):
    '''
    L1距离
    :param y_true:
    :param y_pred:
    :return:
    '''
    return K.mean(K.abs(y_pred - y_true))

def l2_loss(y_true, y_pred):
    '''
    L2距离
    :param y_true:
    :param y_pred:
    :return:
    '''
    return K.mean(K.square(y_pred - y_true))

def bce_loss(y_pred,y_true):
    '''
    Binary_Cross_Error 二分类交叉熵
    :param y_pred:
    :param y_true:
    :return:
    '''
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y_true,logits=y_pred))

def batch_norm(x, name,momentum=0.9,epsilon=1e-5, is_train=True):
    return tf.contrib.layers.batch_norm(x,
         decay=momentum,
         updates_collections=None,
         epsilon=epsilon,scale=True,
         is_training=is_train,
         scope=name)

def Encoder(inputs, opt, istrain=True, name="Encoder_1"):
    assert opt.isize % 16 == 0 , "image_size has to be test_img multiple of 16"

    ''' 初始化层(initial layers) '''
    # 卷积层
    # gf -> num_generator_filter = 64
    # k_h * k_w = 4 * 4
    x = Conv2D(opt.ngf,(4,4),strides=2,padding="same",use_bias=False)(inputs)
    # leaky = 0.2
    x = LeakyReLU(0.2)(x)
    size_now = opt.isize // 2

    ''' 外加层(Extra layers)'''
    for t in range(opt.extralayers):
        x = Conv2D(opt.ngf,(3,3),padding="same",use_bias=False)(x)
        x = batch_norm(x,name+"_BatchNorm_1"+str(t),is_train=istrain)
        x = LeakyReLU(0.2)(x)

    # channel: default number is 64
    channel = opt.ngf

    ''' 减少层(reduction layers) '''
    while size_now > 4:
        x = Conv2D(channel*2,(4,4),strides=2,padding="same",use_bias=False)(x)
        x = batch_norm(x,name+"_BatchNorm2_"+str(channel),is_train=istrain)
        x = LeakyReLU(0.2)(x)
        channel *= 2
        size_now //= 2

    # state size. 64(channel) * 4 * 4
    ''' 输出层(final layers),调整该层的大小为 64(channel)* 1 * 1 '''
    output = Conv2D(opt.nz,(4,4),padding="valid",use_bias=False)(x)

    return output

def Decoder(inputs, opt, istrain=True):
    assert opt.isize % 16 == 0, "image_size has to be test_img multiple of 16"

    # cngf = 32 , tsize = 4
    cngf, tisize = opt.ndf // 2 ,4
    while tisize != opt.isize:
        # after loop, cngf reaches to 512
        cngf *= 2
        tisize *= 2
    ''' z是输入，第一层解卷积层的维度是 channel * 4 * 4 '''
    x = Conv2DTranspose(cngf,(4,4),padding="valid",use_bias=False)(inputs)
    x = batch_norm(x,"BatchNorm1",is_train=istrain)
    x = Activation("relu")(x)

    ''' 增加层(increasing layers)数 '''
    size_now = 4
    while size_now < opt.isize // 2:
        x = Conv2DTranspose(cngf//2,(4,4),strides=2,padding="same",use_bias=False)(x)
        x = batch_norm(x,"BatchNorm_2"+str(size_now),is_train=istrain)
        x = Activation('relu')(x)
        cngf //= 2
        size_now *= 2

    ''' 外加层(extral layers)，保持通道数和层数相同 '''
    for t in range(opt.extralayers):
        x = Conv2DTranspose(cngf,(3,3),padding="same",use_bias=False)(x)
        x = batch_norm(x,"BatchNorm3_"+str(t),is_train=istrain)
        x = Activation("relu")(x)

    ''' 输出层(final layers)，expand the size with 2 and channel of n_output_channer '''
    # nc -> image channel
    x = Conv2DTranspose(opt.nc,(4,4),strides=2,padding="same",use_bias=False)(x)
    x = Activation("tanh")(x)

    return x

def Net_Generator(inputs,opt,reuse=False,istrain=True):
    '''
    生成器网络 :  Encoder -> Decoder -> Encoder 结构
    :param inputs: 原图tensor
    :param opt: 训练参数对象
    :param istrain: 是否训练
    :return: x_star = 重建图
             z = 原图隐空间向量(latent_i_z)
             z_star = 重建图隐空间向量(latent_o_z_star)
    '''
    with tf.variable_scope("Net_Gen"):
        z = Encoder(inputs,opt,istrain=istrain,name="Encoder_1")
        x_star = Decoder(z,opt,istrain=istrain)
        z_star = Encoder(x_star,opt,istrain=istrain,name="Encoder_2")
    return x_star,z,z_star


def Net_Discriminator(inputs,opt,reuse=False,istrain=True,name="Net_Dis"):
    '''
    判别器网络 => 与去掉输出层的Encoder结构类似
            => 结构模型来自于DCGan
            => 最后一层用于原始代码中的特征映射丢失
    :param inputs: 图片tensor
    :param opt: 训练参数对象
    :param reuse: 是否重用
    :param istrain: 是否训练
    :param name:
    :return: feature 输入图的特征向量
              classifier 全连接层的分类器，用于判别fake or real
    '''
    with tf.variable_scope("Net_Dis",reuse=reuse):
        ''' 初始化层(initial layer)'''
        x = Conv2D(opt.ngf,(4,4),strides=2,padding="same",use_bias=False)(inputs)
        x = LeakyReLU(0.2)(x)
        size_now = opt.isize // 2

        ''' 外加层(extra layers)'''
        for t in range(opt.extralayers):
            x = Conv2D(opt.ngf,(3,3),padding="same",use_bias=False)(x)
            x = batch_norm(x,name+"_BatchNorm1_"+str(t),is_train=istrain)
            x = LeakyReLU(0.2)(x)

        channel = opt.ngf

        ''' 减少层(reduction layers) '''
        while size_now > 4:
            x = Conv2D(channel*2,(4,4),strides=2,padding="same",use_bias=False)(x)
            x = batch_norm(x,name+"_BatchNorm2_"+str(channel),is_train=istrain)
            x = LeakyReLU(0.2)(x)
            channel *=  2
            size_now //= 2

        feature = x
        # state size. channel * 4 * 4
        ''' 输出层(final layer) 调整层维度数为 channel * 1 * 1'''
        x = Conv2D(1,(4,4),padding="valid",use_bias=False)(x)
        # 将多维的输入一维化，用于全连接层
        x = Flatten()(x)
        # 全连接层 units = 1
        classifier = Dense(1)(x)

        return feature,classifier



class Ganomaly(object):
    '''Ganomaly类
            :param sess:
            :param opt:
            :param dataloader:
            '''
    @staticmethod
    def name():
        """Return name of the class.
        """
        return 'Ganomaly'
    def __init__(self,sess,opt):
        # tf会话对象
        self.sess = sess
        # training参数对象
        self.opt = opt
        self.is_train = tf.placeholder(tf.bool)
        # 图片大小
        self.imsize = opt.isize
        # 图片维度数
        self.img_shape = [opt.batchsize,opt.isize,opt.isize,3]
        # tf类型的图片tensor
        self.img_input = tf.placeholder(tf.float32,self.img_shape)

        ''' Step0 => build model '''
        print("\nStep0 => build model\n")
        with tf.variable_scope(tf.get_variable_scope(),reuse=tf.AUTO_REUSE):
            # img_gen => 重建图
            # latent_z => 原图隐空间向量
            # latent_z_gen => 重建图隐空间向量
            self.img_gen,self.latent_z,self.latent_z_gen = Net_Generator(self.img_input,self.opt,self.is_train)
            # 将重建图送入判别器返回
            # feature_fake => 重建图特征向量
            # label_fake => 重建图的分类标签
            self.feature_fake,self.label_fake = Net_Discriminator(self.img_gen,self.opt,False,self.is_train)
            # 将原图送入判别器返回
            # feature_real => 原图特征向量
            # label_real => 原图的分类标签
            self.feature_real,self.label_real = Net_Discriminator(self.img_input,self.opt,True,self.is_train)

        # 需要的训练的变量列表
        self.t_vars = tf.trainable_variables()
        # 判别器的训练变量列表
        self.d_vars = [var for var in self.t_vars if "Net_Dis" in var.name]
        # 生成器的训练变量列表
        self.g_vars = [var for var in self.t_vars if "Net_Gen" in var.name]

        ''' Step1 =>  crate losses '''
        print("\nStep1 =>  crate losses\n")
        # 对抗损失(adversarial loss)
        # => 计算判别器输出的原图与重建图的特征向量的L2距离
        # || f(x) - f(x~) ||2
        self.adv_loss = l2_loss(self.feature_fake,self.feature_real)
        # 上下文损失(context loss)
        # => 原图与重建图之间的L1距离
        # || x - x~ ||
        self.context_loss = l1_loss(self.img_input,self.img_gen)
        # 编码器损失(encoder loss)
        # => 生成器返回的原图与重建图的隐空间向量的L2距离
        self.encoder_loss = l2_loss(self.latent_z,self.latent_z_gen)
        # 生成器损失
        # L = Ladv + λLcon + Lenc
        self.generator_loss = self.adv_loss + 50*self.context_loss + self.encoder_loss
        ''' 判别器损失 => real label 趋近于 1，fake label 趋近于 0 '''
        # 判别为真损失
        self.real_loss = bce_loss(self.label_real,tf.ones_like(self.label_real))
        # 判别为假损失
        self.fake_loss = bce_loss(self.label_fake,tf.zeros_like(self.label_fake))
        # 特征向量损失(feature loss)
        self.feature_loss = self.real_loss + self.fake_loss
        # 判别器损失(discriminator loss)
        # => L = bce(real) + bce(fake)
        self.discriminator_loss = self.feature_loss

        ''' Step2 => Optimize the loss,learning rate and beta1 is from Original code of Pytorch '''
        print("\nOptimize the loss\n")
        # 更新变量列表
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.variable_scope(tf.get_variable_scope(),reuse=None):
                self.gen_train_op = tf.train.AdamOptimizer(
                    learning_rate=2e-3,beta1=0.5,beta2=0.999).minimize(self.generator_loss,var_list=self.g_vars)
                self.dis_train_op = tf.train.AdamOptimizer(
                    learning_rate=2e-3,beta1=0.5,beta2=0.999).minimize(self.discriminator_loss,var_list=self.d_vars)

        ''' Step3 => save the model '''
        self.saver = tf.train.Saver()
        print("\nStep3 => save the model\n")

        ''' Step4 => initialization '''
        print("\nStep4 => initialization\n")
        self.sess.run(tf.global_variables_initializer())

    def gen_fit(self, batch_x):
        _, loss,al,cl,el  = self.sess.run([self.gen_train_op,
                                  self.generator_loss,
                                  self.adv_loss,
                                  self.context_loss,
                                  self.encoder_loss],
                               {self.img_input:batch_x,self.is_train: True,})
        return loss,al,cl,el

    def dis_fit(self, batch_x):
        _, loss, dis_real_loss, dis_fake_loss = self.sess.run([self.dis_train_op, self.discriminator_loss,
                                                               self.real_loss,
                                                               self.fake_loss],
                                                              {self.img_input: batch_x, self.is_train: True, })
        return loss, dis_real_loss, dis_fake_loss

    def train(self,batch_x):

        gen_loss,al,cl,el = self.gen_fit(batch_x)
        _,dis_real_loss,dis_fake_loss = self.dis_fit(batch_x)

        #如果判别器损失D_loss为0，就重新初始化Net_Dis
        if dis_real_loss < 1e-5 or dis_fake_loss < 1e-5:
            init_op = tf.initialize_variables(self.d_vars)
            self.sess.run(init_op)
            print("\nRe-Initialize-Net_Discirminator\n")

        return gen_loss,al,dis_real_loss,dis_fake_loss

    def test(self,opt,x_test):
        '''
        测试生成器
        :param x_test: 测试数据
        :return:
        '''
        # saver = tf.train.Saver()
        # with tf.Session() as sess:
        #     sess.run(tf.initialize_all_variables())
        #     saver.restore(sess,tf.train.latest_checkpoint('checkpoint'))
        #     # print(sess.run('Net_Gen:0'))
        #     save_images(x_test, [8, 8], 'output/reals_1.png')
        #     img_gen,_,_ = Net_Generator(self.img_input,self.opt,istrain=False)
        #     gen_imgs = self.sess.run(self.img_gen, {self.img_input: x_test, self.is_train: False})
        #     save_images(gen_imgs, [8, 8], 'output/fakes_1.png')

    def evaluate(self,whole_x,whole_y):
        bs = self.opt.test_batch_size
        labels_out,scores_out = [],[]
        index = 1
        for index in range(int(whole_x.shape[0] / bs)):
            batch_x = whole_x[index*bs:(index+1)*bs]
            batch_y = whole_y[index*bs:(index+1)*bs]
            latent_loss,latent_gen_loss = self.sess.run([self.latent_z,
                                                         self.latent_z_gen],
                                {self.img_input:batch_x,self.is_train:False})
            latent_error = np.mean(abs(latent_loss - latent_gen_loss),axis=-1)
            latent_error = np.reshape(latent_error,[-1])
            scores_out = np.append(scores_out,latent_error)
            labels_out = np.append(labels_out,batch_y)

            ''' 将分数向量缩减到[0,1]之间 '''
            # scores_out = (scores_out - scores_out.min()) / (scores_out.max() - scores_out.min())
        '''  计算ROC的值 '''
        auc_out = roc(labels_out,scores_out)

        return scores_out,labels_out,auc_out

    def save(self,dir_path,i):
        self.saver.save(self.sess,dir_path+"/test_{:02d}.ckpt".format(i))


    # def batch_img_save(self,batch_x,path):

    '''show the generated images'''

    def show(self, single_x):
        generated_img = self.sess.run(self.img_gen, {self.img_input: single_x, self.is_train: False})
        plt.imshow(generated_img[0, :, :, 0])
        plt.show()
        plt.imshow(single_x[0, :, :, 0])
        plt.show()
        return generated_img[0, :, :, 0], single_x[0, :, :, 0]

    def get_gen_img(self,single_x):
        '''
        # 获取重建图
        :param single_x:
        :return:
        '''
        gen_img = self.sess.run(self.img_gen,{self.img_input:single_x,self.is_train:False})

        return gen_img

    def get_dis_feature(self, single_x):
        real_feature,real_label = self.sess.run([self.feature_real,self.label_real],
                                                {self.img_input:single_x,self.is_train:False})
        return real_feature,real_label


    def get_whole_dis(self,whole_x,whole_y):
        bs = self.opt.test_batch_size
        f, l = [], []
        index = 1
        ran = int(whole_x.shape[0] / bs)
        print(ran)
        for index in range(ran):
            batch_x = whole_x[index * bs:(index + 1) * bs]
            batch_y = whole_y[index * bs:(index + 1) * bs]
            feature, label = self.sess.run([self.feature_real, self.label_real],
                                                     {self.img_input: batch_x, self.is_train: False})
            for i, x in enumerate(feature):
                x = np.reshape(x, (4096))
                f.append(x)
                l.append(batch_y[i])
            if index%1 == 0: print(index)
        print("done.")

        f = np.array(f).astype(np.float32)
        l = np.array(l).astype(np.float32)
        return f,l
