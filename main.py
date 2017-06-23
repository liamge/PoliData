import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tf_model import FFNN
from utils import DataLoader, Config
from sklearn.metrics import f1_score, accuracy_score


def train_model(model, num_epochs):

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))

        train_losses = []
        val_losses = []

        for i in range(num_epochs):
            # Main training loop
            epoch_loss = 0.0
            j = 0
            for batch_X, batch_y in model.data.generate_batches(random_state=42, method='train'):
                feed_dict = {model.train_input: batch_X, model.train_labels: batch_y}
                loss, _ = sess.run([model.loss, model.optimizer], feed_dict=feed_dict)
                epoch_loss += loss
                train_losses.append(loss)
                j += 1
                if j % 100 == 0:
                    print("Average batch loss: {}".format(epoch_loss / j))
            # Validation
            val_loss = 0.0
            f1s = 0.0
            accuracies = 0.0
            j = 0
            for batch_X, batch_y in model.data.generate_batches(random_state=42, method='test'):
                feed_dict = {model.train_input: batch_X, model.train_labels: batch_y}
                loss, logits = sess.run([model.loss, model.logits], feed_dict=feed_dict)
                preds = sess.run(tf.nn.softmax(logits))
                preds = np.argmax(preds, axis=1)
                accuracy = accuracy_score(batch_y, preds)
                accuracies += accuracy
                f1 = f1_score(batch_y, preds)
                f1s += f1
                val_loss += loss
                val_losses.append(loss)
                j += 1

            print("Validation loss: {}".format(val_loss / j))
            print("Validation F1: {}".format(f1s / j))
            print("Validation accuracy: {}".format(accuracies / j))
            print("Average loss for epoch {}: {}".format(i, epoch_loss / model.data.max_batches))
            save_path = saver.save(sess, "checkpoints/model-epoch-{}.ckpt".format(i))
            print("Model saved in file: %s" % save_path)

    #plt.plot(train_losses, label='train loss', c='r--')
    plt.plot(val_losses, label='val loss', c='b-')
    plt.show()


if __name__ == '__main__':
    conf = Config()
    dl = DataLoader('full_data.csv', batch_size=conf.batch_size)
    model = FFNN(conf, dl)
    model.build_graph()

    train_model(model, 5)