import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
import os
import network
from get_data import batch_set
from network import conv_layers, fc_layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    train_db, y, test_db, y_test = batch_set()
    sample = next(iter(train_db))
    print(sample[0].shape,sample[1].shape)
    conv_net = Sequential(conv_layers)
    conv_net.build(input_shape=[None, 40, 120, 3])
    fc_net=Sequential(fc_layers)
    fc_net.build(input_shape=[None, 1024])
    optimizer = optimizers.Adam(lr=1e-4)
    # [1,2]+[3,4] = [1,2,3,4]
    variables = conv_net.trainable_variables + fc_net.trainable_variables

    for epoch in range(3):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                # [b,120,40,3]
                out = conv_net(x)
                # flatten ==> [b,512]
                out = tf.reshape(out, [-1, 512])
                # [b,512] --> [b,100]
                logits = fc_net(out)
                # [b] --> [b,100]
                y_onehot = tf.one_hot(y, depth=100)
                # compute loss
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))

        total_num = 0
        total_correct = 0
        for x, y in test_db:
            out = conv_net(x)
            out = tf.reshape(out, [-1, 1024])
            logits = fc_net(out)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += int(correct)

        acc = total_correct / total_num
        print(epoch, 'acc:', acc)


if __name__ == '__main__':
    main()
