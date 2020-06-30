
import numpy as np

import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from scipy.stats import norm
from scipy.stats import t
import matplotlib.pyplot as plt
from scipy import stats, integrate
import pandas as pd
from sklearn.manifold import TSNE

# print('比较t-分布与标准正态分布')



pc = '#3f7f93'
nc = '#c3553a'

# la_i,la_o = [],[]
# for i,ep in enumerate(p_i):
#     la_i.append(ep)
# for i,ep in enumerate(n_i_latent):
#     la_i.append(ep)

# for i,ep in enumerate(p_o_latent):
#     la_o.append(ep)
# for i,ep in enumerate(n_o_latent):
#     la_o.append(ep)


def pdf(p, name,color,color2):
    mu = np.mean(p)  # 计算均值
    sigma = np.std(p)
    num_bins = 100
    n_s, bins, patches = plt.hist(p, num_bins, normed=1, facecolor=color, alpha=0.6)
    # 直方图函数，x为x轴的值，normed=1表示为概率密度，即和为一，绿色方块，色深参数0.5.返回n个概率，直方块左边线的x值，及各个方块对象
    y = scipy.stats.norm.pdf(bins, mu, sigma)  # 拟合一条最佳正态分布曲线y
    plt.plot(bins, y, 'r--',c=color2)  # 绘制y的曲线
    plt.xlabel('sepal-length')  # 绘制x轴
    plt.ylabel('Probability')  # 绘制y轴
    plt.title(r'Histogram : $\mu='+str(mu)+'$,$\sigma='+str(sigma)+'$')  # 中文标题 u'xxx'
    plt.subplots_adjust(left=0.15)  # 左边距
    # plt.show()
    pdf = PdfPages('data/test/pdf/' + name)
    pdf.savefig()
    # plt.close()
    pdf.close()


# pdf(p_i,"p_i.pdf",'blue','blue')
# pdf(p_o,"p_o.pdf",'red','red')
def latent():
    p_i = np.loadtxt("data/test/new_la/n_i.txt")
    p_o = np.loadtxt("data/test/new_la/n_o.txt")
    #
    n_i = np.loadtxt("data/test/new_la/p_i.txt")
    n_o = np.loadtxt("data/test/new_la/p_o.txt")

    p_i = np.array(p_i).astype(np.float32)
    p_i = np.reshape(p_i, 32000)

    p_o = np.array(p_o).astype(np.float32)
    p_o = np.reshape(p_o, 32000)

    n_i = np.array(n_i).astype(np.float32)
    n_i = np.reshape(n_i, 32000)

    n_o = np.array(n_o).astype(np.float32)
    n_o = np.reshape(n_o, 32000)
    global i, df
    sns.set(color_codes=True)
    label, x, x1 = [], [], []
    for i in range(32000):
        x1.append('pos_real')
        label.append('real')
    for i in range(32000):
        x1.append('pos_fake')
        label.append('fake')
    for i in range(32000):
        x1.append('neg_real')
    for i in range(32000):
        x1.append('neg_fake')
    for i in range(64000):
        x.append('pos')
    for i in range(64000):
        x.append('neg')
    label = np.append(label, label)
    # label = np.append(np.zeros(32000),np.ones(32000))
    p = np.append(p_i, p_o)
    n = np.append(n_i, n_o)
    data = []
    data = np.append(p, n)
    # o = []
    # o = np.append(p_o,n_o)
    #
    df = pd.DataFrame({'x1': x1, 'x': x, 'data': data, 'label': label})

    sns.boxplot(x="x1", y="data", data=df)

def io():
    n_i = np.loadtxt("data/txt/n_i.txt").reshape(32000)
    n_o = np.loadtxt("data/txt/n_o.txt").reshape(32000)
    p_i = np.loadtxt("data/txt/p_i.txt").reshape(32000)
    p_o = np.loadtxt("data/txt/p_o.txt").reshape(32000)

    pc = '#3f7f93'
    nc = '#c3553a'
    plim=5
    nlim=3
    sns.set(style="darkgrid")

    p = pd.DataFrame({'x1': p_i, 'x2': p_o})
    n = pd.DataFrame({'x1': n_i, 'x2': n_o})
    # sns.kdeplot(p_i,p_o,shade=True, shade_lowest=False,color=pc)
    # sns.kdeplot(n_i, n_o, shade=True, shade_lowest=False,color=nc)
    # sns.kdeplot('x1', 'x2', data=n)
    # sns.jointplot('x1', 'x2', kind="kde", xlim=(-plim, plim), ylim=(-plim, plim), color=pc, data=p, shade=True, shade_lowest=False)
    # sns.jointplot('x1','x2', kind="kde", xlim=(-nlim, nlim), ylim=(-nlim, nlim),color=nc,data=n,shade=True, shade_lowest=False)
    #
    # sns.set()
    # sns.lmplot(x='x2', y='x1', hue='y', truncate=True, data=d, palette=sns.diverging_palette(220, 20, n=2))
def err():
    # p_i = np.loadtxt("data/txt/p_i.txt")
    # p_o = np.loadtxt("data/txt/p_o.txt")
    # p_i = np.loadtxt("data/txt/n_i.txt")
    # p_o = np.loadtxt("data/txt/n_o.txt")
    # n_e = np.loadtxt("data/txt/n_e.txt")
    # p_e = np.loadtxt("data/txt/p_e.txt")
    n = np.loadtxt("data/test/pdf/n_error.txt")
    p = np.loadtxt("data/test/pdf/p_error.txt")
    e = np.append(p[0:320], n[0:320])

    # e= np.append(p_e[0:320],n_e[0:320])
    x,y = [],[]
    for i in range(640):
        x.append(2)
        y.append(i)
    label =np.append(np.ones(320),np.zeros(320))
    d = pd.DataFrame({'x':x,'e':e,'label':label,'y':y})
    sns.set(style="darkgrid")
    sns.lmplot(x='y',y='e',hue='label',data=d,truncate=True,palette=sns.diverging_palette(220, 20, n=2))
    # sns.jointplot(p_e,n_e,kind='kde',shade=True, shade_lowest=False)
    # sns.violinplot(y='e', x='x', hue='label',data=d, split=True,palette=sns.diverging_palette(220, 20, n=2)).set(ylim=(5,100))
    # sns.violinplot(y='e',x='label',data=d,palette=sns.diverging_palette(220, 20, n=2))
    # sns.boxplot(n_e)
    # sns.distplot(n_e,bins=320,color=pc).set(xlim=(5,25))
    # sns.distplot(p_e,bins=320, color=nc).set(xlim=(5,100))
    # p_e = []
    # for i in range(320):
    #     mean = np.mean(abs(p_i[i] - p_o[i]), axis=-1)
    #     print(mean)
    #     p_e = np.append(p_e,mean*100)
    # np.savetxt("data/txt/new_n_e.txt",p_e)
    # p_e = np.loadtxt("data/txt/p_e.txt")
    # n_e = np.loadtxt("data/txt/n_e.txt")
    # pc = '#3f7f93'
    # nc = '#c3553a'
    # e = np.append(p_e,n_e)
    # p = pd.DataFrame({'x1': p_i, 'x2': p_o})
    #
    # sns.set(style="darkgrid")



def fea():
    x = np.loadtxt("data/txt/t199_x_2.txt")
    y = np.loadtxt("data/txt/t199_y.txt")

    n1 = np.reshape(x[0:320,0],320)
    n2 = np.reshape(x[0:320, 1], 320)
    p1 = np.reshape(x[320:640,0],320)
    p2 = np.reshape(x[320:640, 1], 320)
    y1 = y[0:320]
    y2 = y[320:640]
    d = pd.DataFrame({'x1':x[:,0],'x2':x[:,1],'y':y})
    df = pd.DataFrame({'n1':n1,'n2':n2,'p1':p1,'p2':p2,'y1':y1,'y2':y2})

    sns.set()
    cmap3= sns.diverging_palette(145,280, s=85, l=25, n=7, as_cmap=True)
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    cmap2 = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, as_cmap=True)
    sns.kdeplot(n1,n2,shade=True, shade_lowest=False,color=nc)
    sns.kdeplot(p1, p2, shade=True, shade_lowest=False,cmap='GnBu')
    # sns.palplot(sns.diverging_palette(220, 20, n=7))

    # g = sns.lmplot(x='x2',y='x1',hue='y', truncate=True,data=d,palette=sns.diverging_palette(220, 20, n=2))
    # g.set_axis_labels("Sepal length (mm)", "Sepal width (mm)")
    # print(df['n1'])
    # f, ax = plt.subplots(figsize=(8, 8))
    # ax.set_aspect("equal")

    # ax = sns.kdeplot(n1,n2,cmap="Reds", shade=True, shade_lowest=False,alpah=.8)
    # ax = sns.kdeplot(p1,p2,cmap="Blues", shade=True, shade_lowest=False,alpah=.8)
    #
    # red = sns.color_palette("Reds")[-2]
    # blue = sns.color_palette("Blues")[-2]
    # ax.text(2.5, 8.2, "reals", size=16, color=blue)
    # ax.text(3.8, 4.5, "fakes", size=16, color=red)
def detect_err():
    # n_i = np.loadtxt("data/txt/p_i.txt")
    # p_i = np.loadtxt("data/txt/n_i.txt")
    n_e = np.loadtxt("data/txt/n_e.txt")
    p_e = np.loadtxt("data/txt/p_e.txt")
    # print(n_e)
    # a = np.loadtxt('data/txt/a.txt')
    #
    # b = np.loadtxt('data/txt/b.txt')
    #
    # aa = np.append(a[:,0],a[:,0])
    # bb = np.append(a[:,1],b[:,1])
    #
    # label = np.append(np.zeros(320),np.ones(320))
    d = pd.DataFrame({'x':n_e, 'y': p_e})

    sns.lmplot(x='y', y='x',data=d)
    plt.show()

    # tsne = TSNE(n_components=2, init='pca', random_state=0)

    # print(d['x'])
    # sns.lmplot(x='x',y='x',data=d)
    # plt.show()
    # print(p_i[0])
    # np.ar
    # a = p_i[0]
    # a.reshape(-1, 1)
    # a = tsne.fit_transform(n_i)
    # np.savetxt("data/txt/b.txt",a)
    # print()
    # for i in range(len(p_i)):
        # X = tsne.fit_transform(p_i)


    # p_e = np.loadtxt("data/txt/n_e.txt")
    # std = np.std(p_e)
    # mean = np.mean(p_e)
    # Q1 = np.percentile(p_e, 25)
    # Q3 = np.percentile(p_e, 75)
    #
    # IQR = Q3 - Q1
    # outlier_step = 1.5 * IQR
    # print(Q1, Q3,Q3+outlier_step)

    # new_p = []
    # for i in range(len(p_e)):
    #     # print(i, p_e[i])
    #     if p_e[i] > Q3 + outlier_step:
    #         print(i,p_e[i])
    #     else:new_p.append(p_e[i])

    #
    # mean = np.mean(new_p)
    # std = np.std(new_p)
    # Q3 = np.percentile(new_p, 75)
    # Q1 = 1.5*(np.percentile(new_p, 75) - np.percentile(new_p, 25))
    # # print(std,mean)
    # print(len(new_p),mean,std,Q1,Q3)
    # print(outlier_list_col)

def save_pdf(name):
    pdf = PdfPages('data/pdf/'+name+".pdf")
    pdf.savefig()
    pdf.close()
    plt.show()

detect_err()

# err()
# io()
# fea()
# name = "e_e"
# save_pdf(name)

