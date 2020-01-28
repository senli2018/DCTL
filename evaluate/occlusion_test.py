import tensorflow as tf
from our_model.model import CycleGAN
from lib.reader_image import get_test_batch2
from datetime import datetime
import os
import logging
import cv2
import numpy as np
from evalute.plot_til.plot import draw_heatmap
from lib import utils

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_integer('image_size', 256, 'image size, default: 256')
tf.flags.DEFINE_bool('use_lsgan', True,
                     'use lsgan (mean squared error) or cross entropy loss, default: True')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')
tf.flags.DEFINE_integer('lambda1', 10,
                        'weight for forward cycle loss (X->Y->X), default: 10')
tf.flags.DEFINE_integer('lambda2', 10,
                        'weight for backward cycle loss (Y->X->Y), default: 10')
tf.flags.DEFINE_float('learning_rate', 2e-4,
                      'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5,
                      'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('pool_size', 50,
                      'size of image buffer that stores previously generated images, default: 50')
tf.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_string('load_model', None,
                       'folder of saved our_model that you wish to continue training (e.g. 20170602-1936), default: None')

model = 'train'  # learning_loss_set = 4. train or test
Uy = None


def train():
    if FLAGS.load_model is not None:
        checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
    else:
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoints_dir = "checkpoints/{}".format(current_time)
        try:
            os.makedirs(checkpoints_dir)
        except os.error:
            pass

    graph = tf.Graph()
    with graph.as_default():
        cycle_gan = CycleGAN(
            batch_size=FLAGS.batch_size,
            image_size=FLAGS.image_size,
            use_lsgan=FLAGS.use_lsgan,
            norm=FLAGS.norm,
            lambda1=FLAGS.lambda1,
            lambda2=FLAGS.lambda2,
            learning_rate=FLAGS.learning_rate,
            beta1=FLAGS.beta1,
            ngf=FLAGS.ngf
        )

        G_loss, D_Y_loss, F_loss, D_X_loss, teacher_loss, student_loss, learning_loss, \
        x_correct, y_correct, fake_x_correct, softmax3, fake_x_pre, f_fakeX, fake_x, fake_y = cycle_gan.model()
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        if FLAGS.load_model is not None:
            checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            meta_graph_path = "checkpoints/20190611-1650/our_model.ckpt-30000.meta"
            print('meta_graph_path', meta_graph_path)
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, "checkpoints/20190611-1650/our_model.ckpt-30000")

            step = 0

        else:
            sess.run(tf.global_variables_initializer())
            step = 0

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                result_dir = './result'
                fake_dir = os.path.join(result_dir, 'fake_xy')
                roc_dir = os.path.join(result_dir, 'roc_curves')
                plot_dir = os.path.join(result_dir, 'tsne_pca')
                conv_dir = os.path.join(result_dir, 'convs')
                occ_dir = os.path.join(result_dir, 'occ_test')
                Xconv_dir = os.path.join(result_dir, 'Xconv_dir')
                fakeXconv_dir = os.path.join(result_dir, 'fakeXconv_dir')
                Y_VGGconv_dir = os.path.join(result_dir, 'Y_VGGconv_dir')
                fakeY_VGGconv_dir = os.path.join(result_dir, 'fakeY_VGGconv_dir')

                rconv_dir = os.path.join(result_dir, 'resconvs')
                utils.prepare_dir(result_dir)
                utils.prepare_dir(occ_dir)
                utils.prepare_dir(fake_dir)
                utils.prepare_dir(roc_dir)
                utils.prepare_dir(plot_dir)
                utils.prepare_dir(conv_dir)
                utils.prepare_dir(rconv_dir)
                utils.prepare_dir(Xconv_dir)
                utils.prepare_dir(fakeXconv_dir)
                utils.prepare_dir(Y_VGGconv_dir)
                utils.prepare_dir(fakeY_VGGconv_dir)

                x_image, x_label, oximage = get_test_batch2("X", 1, FLAGS.image_size, FLAGS.image_size, "./dataset/")
                y_image, y_label, oyimage = get_test_batch2("Y", 1, FLAGS.image_size, FLAGS.image_size, "./dataset/")

                image = y_image[1]
                width = height = 256
                occluded_size = 16
                data = np.empty((width * height + 1, width, height, 3), dtype="float32")
                data[0, :, :, :] = image
                cnt = 1
                for i in range(height):
                    for j in range(width):
                        i_min = int(i - occluded_size / 2)
                        i_max = int(i + occluded_size / 2)
                        j_min = int(j - occluded_size / 2)
                        j_max = int(j + occluded_size / 2)
                        if i_min < 0:
                            i_min = 0
                        if i_max > height:
                            i_max = height
                        if j_min < 0:
                            j_min = 0
                        if j_max > width:
                            j_max = width
                        data[cnt, :, :, :] = image
                        data[cnt, i_min:i_max, j_min:j_max, :] = 255
                        cnt += 1

                u_ys = np.empty([data.shape[0]], dtype='float64')
                occ_map = np.empty((width, height), dtype='float64')

                print('occ_map.shape', occ_map.shape)
                cnt = 0
                feature_y_eval = sess.run(
                    softmax3,
                    feed_dict={cycle_gan.y: [data[0]]})  #


                u_y0 = feature_y_eval[0]
                [idx_u] = np.where(np.max(u_y0))
                idx_u = idx_u[0]
                print('feature_y_eval', feature_y_eval)
                print('u_y0', u_y0)
                max = 0
                print('len u_y0', len(u_y0))
                for val in range(len(u_y0)):
                    vallist = u_y0[val]
                    if vallist > max:
                        max = vallist

                u_y0 = max
                print('max', u_y0)

                for i in range(width):
                    for j in range(height):
                        feature_y_eval = sess.run(
                            softmax3,
                            feed_dict={cycle_gan.y: [data[cnt + 1]]})
                        u_y = feature_y_eval[0]
                        print('u_y', u_y)
                        u_y1 = 0
                        for val in range(len(u_y)):
                            vallist = u_y[val]
                            if vallist > u_y1:
                                u_y1 = vallist

                        occ_value = u_y0 - u_y1
                        occ_map[i, j] = occ_value
                        print(str(cnt) + ':' + str(occ_value))
                        cnt += 1

                occ_map_path = os.path.join(occ_dir, 'occlusion_map_{}.txt'.format('1'))
                np.savetxt(occ_map_path, occ_map, fmt='%0.8f')
                cv2.imwrite(os.path.join(occ_dir, '{}.png'.format('1')), oyimage[1])
                draw_heatmap(occ_map_path=occ_map_path, ori_img=oyimage[1],
                             save_dir=os.path.join(occ_dir, 'heatmap_{}.png'.format('1')))








        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:

            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)

def main(unused_argv):
    train()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
