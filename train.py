import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
from get_data import *
from network import conv_layers, fc_layers

# pic_train_orig_x = "data/train"     #
pic_train_orig_y = "data/train_label.csv"   # csv读取路径，应该有所有的图的y
# pic_wanna_orig = "data/test"                #
pic_train_del = "data/del"                  # 图片读取路径

# 输入 测试的 （x，y）/卷积网络层参数/全链接网络层参数/循环轮回数/？？？
def predict(db, conv_net, fc_net, epoch, acclist):
    total_num = 0
    total_correct = 0
    for x, y in db:
        out = conv_net(x)
        logits = fc_net(out)
        prob = tf.nn.softmax(logits, axis=1)
        pred = tf.argmax(prob, axis=1)                          #输出单个的自定义ascii的代码【1，3，5】
        pred = tf.cast(pred, dtype=tf.int32)
        correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)    #对比，相同为1【1，1，0】
        correct = tf.reduce_sum(correct)                        #计算多少个1，即多少个成功预测
        total_num += x.shape[0]
        total_correct += int(correct)
    acc = total_correct / total_num
    print(epoch, 'acc:', acc)                                   #epoch只为打印是第多少个轮回
    acclist.append(acc)
    print(acclist)                                              #打印到现在为止每次的准星列表
    return acc


def main():
    #加载数据
    #函数在 get_dataset.py
    print("读取数据ing")
    train_db, test_db = load_datasets()
    # 实体化网络，具体在network.py
    print("加载网络ing")
    conv_net = Sequential(conv_layers)
    conv_net.build(input_shape=[None, 40, 25, 3])
    fc_net = Sequential(fc_layers)
    fc_net.build(input_shape=[None, 1024])
    optimizer = optimizers.Adam(lr=1e-5)
    variables = conv_net.trainable_variables + fc_net.trainable_variables
    # acctrain存储之前每次准星，acctest相似
    print("开始训练")
    acctrain = []
    acctest = []
    for epoch in range(200):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = conv_net(x)
                logits = fc_net(out)
                y_onehot = tf.one_hot(y, depth=62)
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))
            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))

        print('train:')
        new_train_acc = predict(train_db, conv_net, fc_net, epoch, acctrain)

        print('test:')
        new_test_acc = predict(test_db, conv_net, fc_net, epoch, acctest)
        # 如果test准星最高，存储一次 h5 文件
        if new_test_acc>max(acctest):
            conv_net.save("model/conv_model.h5")
            fc_net.save("model/fc_model.h5")


if __name__ == '__main__':
    main()
