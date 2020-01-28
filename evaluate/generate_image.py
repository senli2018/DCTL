import tensorflow as tf
from our_model.model import CycleGAN
from lib.reader_image import get_test_batch2
from datetime import datetime
import os
import logging
# from utils import ImagePool
import cv2
import numpy as np
from evalute.plot_til.plot import  plot_conv_output,draw_heatmap
from lib import utils

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
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

tf.flags.DEFINE_string('load_model',None,
                       'folder of saved our_model that you wish to continue training (e.g. 20170602-1936), default: None')

model='train'  #     learning_loss_set = 4. train or test
Uy  =     None
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
        x_correct, y_correct, fake_x_correct, softmax3, fake_x_pre, f_fakeX, fake_x, fake_y= cycle_gan.model()

        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        if FLAGS.load_model is not None:
            checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
            step = int(meta_graph_path.split("-")[2].split(".")[0])
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
                Xconv_dir =  os.path.join(result_dir, 'Xconv_dir')
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

                x_image, x_label,oximage = get_test_batch2("X",40,FLAGS.image_size,FLAGS.image_size,"./dataset/")
                y_image, y_label,oyimage = get_test_batch2("Y",40,FLAGS.image_size,FLAGS.image_size,"./dataset/")


                length3 = len(y_label)
                features_d = []
                fakeX_img =[]
                fakeY_img = []
                resconv_y_eval = []
                vggconv_x_eval = []
                X_Resconv_eval = []
                fakeX_Resconv_eval = []
                Y_VGGconv_eval = []
                fakeY_VGGconv_eval = []



                for i in range(length3):

                    ximgs = []
                    xlbs = []
                    yimgs=[]
                    ylbs=[]

                    ximgs.append(x_image[i])
                    xlbs.append(x_label[i])
                    yimgs.append(y_image[i])
                    ylbs.append(y_label[i])


                    softmax, fake_xpre, ffakeX, fakeX, fakeY ,X_Resconv,fakeX_Resconv , Y_VGGconv,fakeY_VGGconv= (sess.run(
                        [softmax3, fake_x_pre, f_fakeX, fake_x, fake_y, tf.get_collection('X_Resconv'),tf.get_collection('fakeX_Resconv'),tf.get_collection('Y_VGGconv'),tf.get_collection('fakeY_VGGconv')],
                        feed_dict={cycle_gan.x: ximgs, cycle_gan.y: yimgs,
                                   cycle_gan.x_label: xlbs, cycle_gan.y_label: ylbs}

                     )
                    )
                    Uy =  softmax[0]
                    fake_x_img = (np.array(fakeX[0]) + 1.0) * 127.5
                    fake_x_img = cv2.cvtColor(fake_x_img, cv2.COLOR_RGB2BGR)
                    fake_y_img = (np.array(fakeY[0]) + 1.0) * 127.5
                    fake_y_img = cv2.cvtColor(fake_y_img, cv2.COLOR_RGB2BGR)
                    fakeX_img.append(fake_x_img)
                    fakeY_img.append(fake_y_img)

                    X_Resconv_eval.append(X_Resconv)
                    fakeX_Resconv_eval.append(fakeX_Resconv)
                    Y_VGGconv_eval.append(Y_VGGconv)
                    fakeY_VGGconv_eval.append(fakeY_VGGconv)
                    features_d.append(ffakeX[0])

                # Cross Domain Image Generation#
                for i in range(length3):
                    file_nameOX = os.path.join(fake_dir, str(i) + '_oriX.png')
                    cv2.imwrite(file_nameOX, oximage[i])
                    file_name_fakeX = os.path.join(fake_dir, str(i) + '_fakeX.png')
                    cv2.imwrite(file_name_fakeX, fakeX_img[i])
                    file_nameOY = os.path.join(fake_dir, str(i) + '_oriY.png')
                    cv2.imwrite(file_nameOY, oyimage[i])
                    file_name_fakeY = os.path.join(fake_dir, str(i) + '_fakeY.png')
                    cv2.imwrite(file_name_fakeY, fakeY_img[i])

                # Feature Map Visualization  fake_X
                width = height = 256
                vggconv = vggconv_x_eval
                for step in range(length3):


                    id_x_dir = os.path.join(Xconv_dir, str(step))
                    print('id_x_dir', id_x_dir)
                    for i, c in enumerate(X_Resconv_eval[step]):
                        plot_conv_output(c, i, id_x_dir)
                        print('Res%d' %i)
                    cv2.imwrite(os.path.join(id_x_dir, 'X.png'), oximage[step])

                for step in range(length3):
                    id_fakex_dir = os.path.join(fakeXconv_dir, str(step))
                    print('id_fakex_dir', id_fakex_dir)
                    for i, c in enumerate(fakeX_Resconv_eval[step]):
                        plot_conv_output(c, i, id_fakex_dir)
                        print('fakeRes%d' %i)
                    cv2.imwrite(os.path.join(id_fakex_dir, 'fakeX.png'), fakeX_img[step])




                for step in range(length3):
                    id_y_dir = os.path.join(Y_VGGconv_dir, str(step))
                    print('id_y_dir', id_y_dir)
                    for i, c in enumerate(Y_VGGconv_eval[step]):
                        plot_conv_output(c, i, id_y_dir)
                        print('VGG%d' % i)
                    cv2.imwrite(os.path.join(id_y_dir, 'y.png'), oyimage[step])


                for step in range(length3):
                    id_fakey_dir = os.path.join(fakeY_VGGconv_dir, str(step))
                    print('id_fekey_dir', id_fakey_dir)
                    for i, c in enumerate(fakeY_VGGconv_eval[step]):
                        plot_conv_output(c, i, id_fakey_dir)
                        print('fakeVGG%d' % i)
                    cv2.imwrite(os.path.join(id_fakey_dir, 'fakeY.png'), fakeY_img[step])

                image = y_image[1]
                width = height = 256


                # occlustion test
                occluded_size = 16
                data = np.empty((width * height + 1, width, height, 3), dtype="float32")
                print('data  ---')
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
                        # print(data[i].shape)
                        cnt += 1
                #
                # [idx_u]=np.where(np.max(Uy[id_y]))


                # [idx_u]=np.where(np.max(Uy))

                u_ys=np.empty([data.shape[0]],dtype='float64')
                occ_map=np.empty((width,height),dtype='float64')
                
                print('occ_map.shape',occ_map.shape)
                cnt=0
                feature_y_eval = sess.run(
                        softmax3,
                        feed_dict={cycle_gan.y: [data[0]]})#

                # print('softmax3',feature_y_eval.eval())

                
                u_y0 =   feature_y_eval[0]
                [idx_u]=np.where(np.max(u_y0))
                idx_u=idx_u[0]
                print('feature_y_eval',feature_y_eval)
                print('u_y0',u_y0)
                max = 0
                print('len u_y0',len(u_y0))
                for val in range(len(u_y0)):
                    vallist =    u_y0[val]
                    if   vallist> max:
                        max = vallist


                u_y0 = max
                # print('max', u_y0[idx_u])
                print('max', u_y0)
                # print('u_y01',u_y0[idx_u])

                for i in range(width):
                    for j in range(height):
                     feature_y_eval = sess.run(
                        softmax3,
                        feed_dict={cycle_gan.y: [data[cnt+1]]})
                     u_y = feature_y_eval[0]
                     # u_y =  max(u_y)
                     print('u_y',u_y)
                     u_y1 = 0
                     for val in range(len(u_y)):
                         vallist =   u_y[val]
                         if   vallist> u_y1:
                             u_y1 = vallist

                     occ_value=u_y0-u_y1
                     occ_map[i,j]=occ_value
                     print(str(cnt)+':'+str(occ_value))
                     cnt+=1

                occ_map_path=os.path.join(occ_dir,'occlusion_map_{}.txt'.format('1'))
                np.savetxt(occ_map_path, occ_map, fmt='%0.8f')
                cv2.imwrite(os.path.join(occ_dir, '{}.png'.format('1')), oyimage[1])
                draw_heatmap(occ_map_path=occ_map_path,ori_img=oyimage[1],save_dir=os.path.join(occ_dir,'heatmap_{}.png'.format('1')))

                exit()
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
