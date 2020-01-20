import tensorflow as tf
import tensorflow.contrib as cb



def identity_block(input,kernel_size,num ,out_filters,stage,block,trainning=True,weight_decay=0.05,reuse=tf.AUTO_REUSE):
    '''
    input：输入的图像
    kernel_size：卷积核的尺寸
    strides：步长
    in_filters：输入的卷积核的个数
    out_filters：输出的卷积核的个数
    stage：阶段
    block：块
    trainning：是否是训练模式
    weight_decay：正则化
    '''
    block_name = "Resnet50"+str(stage)+block
    f1,f2,f3 = out_filters
    with tf.variable_scope(block_name, reuse=reuse):
        #print(block_name, "id.input", input)
        shortcut = input

        #first
        x = tf.layers.conv2d(input,f1,[1,1],[1,1],
                             kernel_regularizer=cb.layers.l2_regularizer(weight_decay),padding="SAME")
        if num == '1':
            tf.add_to_collection('X_Resconv', x)
        else:
            tf.add_to_collection('fakeX_Resconv', x)
        x = tf.layers.batch_normalization(x,axis=3,training=trainning)
        x = tf.nn.relu(x)
        #print(block_name,"id.first",x)

        #second
        x = tf.layers.conv2d(x, f2, [kernel_size, kernel_size], [1, 1],
                             kernel_regularizer=cb.layers.l2_regularizer(weight_decay), padding="SAME")
        if num == '1':
            tf.add_to_collection('X_Resconv', x)
        else:
            tf.add_to_collection('fakeX_Resconv', x)

        x = tf.layers.batch_normalization(x,axis=3,training=trainning)
        x = tf.nn.relu(x)
        #print(block_name,"id.second", x)

        #third
        x = tf.layers.conv2d(x, f3, [1, 1], [1, 1],
                             kernel_regularizer=cb.layers.l2_regularizer(weight_decay), padding="VALID")
        if num == '1':
            tf.add_to_collection('X_Resconv', x)
        else:
            tf.add_to_collection('fakeX_Resconv', x)

        x = tf.layers.batch_normalization(x, axis=3, training=trainning)
        x = tf.nn.relu(x)
        #print(block_name,"id.third", x)

        #final step
        add = tf.add(x,shortcut )
        add_result = tf.nn.relu(add)
        #print(block_name,"id.final", add)
    return add_result

def convolutional_block(input,kernel_size,num,out_filters,stage,block,trainning=tf.AUTO_REUSE ,reuse=tf.AUTO_REUSE,weight_decay = 0.05,strides=2):
    '''
        input：输入的图像
        kernel_size：卷积核的尺寸
        strides：步长,默认步长为2
        in_filters：输入的卷积核的个数
        out_filters：输出的卷积核的个数
        stage：阶段
        block：块
        trainning：是否是训练模式
        weight_decay：正则化
    '''
    block_name = "Resnet50" + str(stage) + block
    f1, f2, f3 = out_filters
    with tf.variable_scope(block_name, reuse=reuse):
        #print(block_name, "conv.input", input)
        shortcut = input

        #first
        x = tf.layers.conv2d(input,f1,[1,1],[strides,strides],
                             kernel_regularizer=cb.layers.l2_regularizer(weight_decay),padding="VALID")
        if num == '1':
            tf.add_to_collection('X_Resconv', x)
        else:
            tf.add_to_collection('fakeX_Resconv', x)
        x = tf.layers.batch_normalization(x,axis=3,training=trainning)
        x = tf.nn.relu(x)
        #print(block_name,"conv.first", x)

        #second
        x = tf.layers.conv2d(x, f2, [kernel_size, kernel_size], [1, 1],
                             kernel_regularizer=cb.layers.l2_regularizer(weight_decay), padding="SAME")
        if num == '1':
            tf.add_to_collection('X_Resconv', x)
        else:
            tf.add_to_collection('fakeX_Resconv', x)

        x = tf.layers.batch_normalization(x,axis=3,training=trainning)
        x = tf.nn.relu(x)
        #print(block_name,"conv.second", x)

        #third
        x = tf.layers.conv2d(x, f3, [1, 1], [1, 1],
                             kernel_regularizer=cb.layers.l2_regularizer(weight_decay), padding="VALID")
        if num == '1':
            tf.add_to_collection('X_Resconv', x)
        else:
            tf.add_to_collection('fakeX_Resconv', x)
        x = tf.layers.batch_normalization(x, axis=3, training=trainning)
        x = tf.nn.relu(x)
        #print(block_name,"conv.third", x)

        #shortcut
        shortcut = tf.layers.conv2d(shortcut,f3,[1,1],[strides,strides],
                             kernel_regularizer=cb.layers.l2_regularizer(weight_decay),padding="VALID")
        if num == '1':
            tf.add_to_collection('X_Resconv', x)
        else:
            tf.add_to_collection('fakeX_Resconv', x)
       # print(block_name,"conv.shortcut", shortcut)

        #final
        add = tf.add(x,shortcut)
        add_result = tf.nn.relu(add)
        #print(block_name,"conv.add", add)
    return add_result

class resnetClassifier:
    print('Building resnetClassifier')
    def __init__(self, name, is_training, reuse=tf.AUTO_REUSE ):
        self.name = name
        self.is_training = is_training

        self.reuse = tf.AUTO_REUSE

    def __call__(self, input,num):
        with tf.variable_scope(self.name, reuse=self.reuse):
            training = self.is_training
            # stage1
            x = tf.layers.conv2d(input, 64, [7, 7], [2, 2], padding="VALID", reuse=self.reuse)
            if num == '1':
                tf.add_to_collection('X_Resconv', x)
            else:
                tf.add_to_collection('fakeX_Resconv', x)

            # tf.add_to_collection('conv_output', x)
            x = tf.layers.batch_normalization(x, axis=3, training=training)
            x = tf.nn.relu(x)
            x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1], padding="VALID")

            # stage2
            '''
                        input：输入的图像
                        kernel_size：卷积核的尺寸
                        strides：步长,默认步长为2
                        in_filters：输入的卷积核的个数
                        out_filters：输出的卷积核的个数
                        stage：阶段
                        block：块
                        trainning：是否是训练模式
                        weight_decay：正则化
                    '''
            x = convolutional_block(x, 3, num,[64, 64, 256], 2, "a", training, strides=1, reuse=self.reuse)


            x = identity_block(x, 3,  num,[64, 64, 256], 2, "b", training, reuse=self.reuse)
            x = identity_block(x, 3, num, [64, 64, 256], 2, "c", training, reuse=self.reuse)

            # stage3
            x = convolutional_block(x, 3, num, [128, 128, 512], 3, "a", training, reuse=self.reuse)
            x = identity_block(x, 3, num, [128, 128, 512], 3, "b", training, reuse=self.reuse)
            x = identity_block(x, 3,num,  [128, 128, 512], 3, "c", training, reuse=self.reuse)
            x = identity_block(x, 3, num, [128, 128, 512], 3, "d", training, reuse=self.reuse)

            # stage4
            x = convolutional_block(x, 3,num,  [256, 256, 1024], 4, "a", training, reuse=self.reuse)
            x = identity_block(x, 3, num, [256, 256, 1024], 4, "b", training, reuse=self.reuse)
            x = identity_block(x, 3, num, [256, 256, 1024], 4, "c", training, reuse=self.reuse)
            x = identity_block(x, 3,num,  [256, 256, 1024], 4, "d", training, reuse=self.reuse)
            x = identity_block(x, 3, num, [256, 256, 1024], 4, "e", training, reuse=self.reuse)
            x = identity_block(x, 3,num,  [256, 256, 1024], 4, "f", training, reuse=self.reuse)

            # stage5
            x = convolutional_block(x, 3, num, [512, 512, 2048], 5, "a", training, reuse=self.reuse)
            x = identity_block(x, 3, num, [512, 512, 2048], 5, "b", training, reuse=self.reuse)
            x = identity_block(x, 3,num,  [512, 512, 2048], 5, "c", training, reuse=self.reuse)

            x = tf.nn.avg_pool(x, [1, 2, 2, 1], strides=[1, 1, 1, 1], padding="VALID")

            flatten = tf.layers.flatten(x)
            x = tf.layers.dense(flatten,units=50,activation=tf.nn.relu)
            logits = tf.layers.dense(x,units=4)
            softmax = tf.nn.softmax(logits )
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            return logits,softmax
