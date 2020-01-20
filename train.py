import tensorflow as tf
from our_model.model import CycleGAN
from lib.reader_image import get_train_batch,get_test_batch1,get_roc_batch
from datetime import datetime
import os
import logging


os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 32')
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

model='train'  # train or test

def train():
    max_accuracy = 0
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
        x_correct, y_correct, fake_x_correct, softmax3, fake_x_pre, f_fakeX, fake_x, fake_y_= cycle_gan.model()
        optimizers = cycle_gan.optimize(G_loss, D_Y_loss, F_loss, D_X_loss, teacher_loss, student_loss, learning_loss)
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
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

                x_image, x_label = get_train_batch("X",FLAGS.batch_size,FLAGS.image_size,FLAGS.image_size,"./dataset/")

                y_image, y_label = get_train_batch("Y",FLAGS.batch_size,FLAGS.image_size,FLAGS.image_size,"./dataset/")


                _, G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val,teacher_loss_eval, student_loss_eval, learning_loss_eval, summary = (
                    sess.run(
                        [optimizers, G_loss, D_Y_loss, F_loss, D_X_loss, teacher_loss, student_loss, learning_loss, summary_op],
                        feed_dict={cycle_gan.x: x_image, cycle_gan.y: y_image,
                                   cycle_gan.x_label:x_label,cycle_gan.y_label:y_label}
                    )
                )

                train_writer.add_summary(summary, step)
                train_writer.flush()

                print('start trianing')
                if  step % 1 ==0:

                    print('-----------Step %d:-------------' % step)
                    print('  G_loss   : {}'.format(G_loss_val))
                    print('  D_Y_loss : {}'.format(D_Y_loss_val))
                    print('  F_loss   : {}'.format(F_loss_val))
                    print('  D_X_loss vb: {}'.format(D_X_loss_val))
                    print('teacher_loss: {}'.format(teacher_loss_eval))
                    print('student_loss: {}'.format(student_loss_eval))
                    print('learning_loss: {}'.format(learning_loss_eval))

                if step% 10000 == 0 and step>0:
                    print('Now is in testing! Please wait result...')

                    test_images_x,test_labels_x,_= get_test_batch1('X',1000,FLAGS.image_size,FLAGS.image_size,"./dataset/")
                    test_images_y,test_labels_y= get_roc_batch(FLAGS.image_size,FLAGS.image_size,"./dataset/Y")
                    y_correct_cout=0
                    fake_x_correct_cout=0
                    print(len(test_images_y))
                    print(len(test_images_x))
                    for i in range(min(len(test_images_y),len(test_images_x))):
                        y_imgs = []
                        y_lbs = []
                        y_imgs.append(test_images_y[i])
                        y_lbs.append(test_labels_y[i])
                        y_correct_eval, fake_x_correct_eval = (
                        sess.run(
                            [y_correct, fake_x_correct],
                            feed_dict={ cycle_gan.y: y_imgs,
                                        cycle_gan.y_label: y_lbs}
                        )
                    )

                        if y_correct_eval:
                            y_correct_cout=y_correct_cout+1
                        if fake_x_correct_eval:
                            fake_x_correct_cout=fake_x_correct_cout+1

                    print('fake_x_correct_cout',fake_x_correct_cout)
                    print('x_accuracy: {}'.format(y_correct_cout/(min(len(test_labels_y),len(test_labels_x)))))
                    print('fake_x_accuracy: {}'.format(fake_x_correct_cout/(min(len(test_labels_y),len(test_labels_x)))))
                    y_accuracy_1 = format(y_correct_cout / (min(len(test_labels_y), len(test_labels_x))))
                    fake_y_accuracy_1 = format(fake_x_correct_cout / (min(len(test_labels_y), len(test_labels_x))))
                    save_path = saver.save(sess, checkpoints_dir + "/our_model.ckpt", global_step=step)
                    print("Model saved in file: %s" % save_path)


                    if float(fake_y_accuracy_1)>max_accuracy:
                        max_accuracy = float(fake_y_accuracy_1)
                        if not os.path.exists(checkpoints_dir ):
                            os.makedirs(checkpoints_dir )
                        f = open(checkpoints_dir + "/step.txt",'w')
                        f.seek(0)
                        f.truncate()
                        f.write(str(step)+'\n')
                        f.write((fake_y_accuracy_1+'\n'))
                        f.close()
                        save_path = saver.save(sess, checkpoints_dir + "/bestmodel/our_model.ckpt", global_step=step)
                        print("Model saved in file: %s" % save_path)

                step += 1

        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            save_path = saver.save(sess, checkpoints_dir + "/our_model.ckpt", global_step=step)
            print("Model saved in file: %s" % save_path)
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)

def main(unused_argv):
    train()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
