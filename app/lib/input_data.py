import os

import tensorflow as tf
from PIL import Image

from app.lib.model import dataset_files


def create_records(path,name,classes):
    '''
    制作二进制数据集
    :param path:
    :param name:
    :param classes:
    :return:
    '''
    writer = tf.python_io.TFRecordWriter('data/tfrecords/train/train.tfrecords')
    for index, name in enumerate(classes):
        for img_path in dataset_files(os.path.join(path,str(classes[index]))):
            img = Image.open(img_path)
            img = img.resize((32, 32))
            img_raw = img.tobytes()
            print(index, img_raw)
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[classes[index]])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
            writer.write(example.SerializeToString())
    writer.close()

def read_and_decode(filename):
    '''
    读取二进制文件
    :param filename:
    :param is_batch:
    :return:
    '''
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    # 解析符号化样本
    features = tf.parse_single_example(
        serialized_example,
        features={
        'label':tf.FixedLenFeature([],tf.int64),
        'img_raw':tf.FixedLenFeature([],tf.string)
    })
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [32, 32, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(label, tf.int32)

    return img,label


if __name__ == "__main__":
    create_records('data/PV/train','train',[1])
    # create_records('data/PV/n_test', 'test', [0,1])
    # img, label = read_and_decode('data/tfrecords/test/test.tfrecords')
    # img_batch,img_label = tf.train.shuffle_batch([img,label],
    #                                                 batch_size=64, capacity=2000,
    #                                                 min_after_dequeue=1000)
    # init = tf.initialize_all_variables()
    # with tf.Session() as sess:
    #     sess.run(init)
    #     # 启动队列
    #     threads = tf.train.start_queue_runners(sess=sess)
    #     for i in range(5):
    #         print(img_batch.shape)
    #         dadsa,l= sess.run([img_batch,img_label])
    #         # l = to_categorical(l, 12)
    #         print(dadsa.shape)

