import cv2
import numpy as np
import tensorflow as tf
import os
import multiprocessing
import util

class Unet:
    def __init__(self):
        pass

    def _conv(self, x, kernels, istraining):
        x = tf.keras.layers.Conv2D(kernels, (3, 3), padding = "same")(x)
        x = tf.keras.layers.BatchNormalization()(x, training = istraining)
        x = tf.nn.relu(x)
        return x

    def build(self, input_tensor, istraining = True):
        x = self._conv(input_tensor, 64, istraining)
        conv1 = self._conv(x, 64, istraining)
        shape1 = tf.shape(conv1)
        x = tf.keras.layers.MaxPool2D(padding="same")(conv1)

        x = self._conv(x, 64, istraining)
        conv2 = self._conv(x, 64, istraining)
        shape2 = tf.shape(conv2)
        x = tf.keras.layers.MaxPool2D(padding="same")(conv2)

        x = self._conv(x, 64, istraining)
        conv3 = self._conv(x, 64, istraining)
        shape3 = tf.shape(conv3)
        x = tf.keras.layers.MaxPool2D(padding="same")(conv3)

        x = self._conv(x, 64, istraining)
        x = self._conv(x, 64, istraining)
        conv4 = tf.keras.layers.Dropout(0.5)(x)
        shape4 = tf.shape(conv4)
        x = tf.keras.layers.MaxPool2D(padding="same")(conv4)

        x = self._conv(x, 64, istraining)
        x = self._conv(x, 64, istraining)

        x = tf.image.resize_images(x, (shape4[1], shape4[2]), method = tf.image.ResizeMethod.BILINEAR)
        x = tf.keras.layers.Concatenate()([x, conv4])
        x = self._conv(x, 64, istraining)
        x = self._conv(x, 64, istraining)

        x = tf.image.resize_images(x, (shape3[1], shape3[2]), method = tf.image.ResizeMethod.BILINEAR)
        x = tf.keras.layers.Concatenate()([x, conv3])
        x = self._conv(x, 64, istraining)
        x = self._conv(x, 64, istraining)        

        x = tf.image.resize_images(x, (shape2[1], shape2[2]), method = tf.image.ResizeMethod.BILINEAR)
        x = tf.keras.layers.Concatenate()([x, conv2])
        x = self._conv(x, 64, istraining)
        x = self._conv(x, 64, istraining)

        x = tf.image.resize_images(x, (shape1[1], shape1[2]), method = tf.image.ResizeMethod.BILINEAR)
        x = tf.keras.layers.Concatenate()([x, conv1])
        x = self._conv(x, 64, istraining)
        x = self._conv(x, 64, istraining)

        x = tf.keras.layers.Conv2D(1, (3, 3), padding = "same")(x)

        return x

    def compile(self):
        input_tensor = tf.placeholder(tf.float32, [None, None, None, 3])
        label_tensor = tf.placeholder(tf.float32, [None, None, None, 1])

        pred = self.build(input_tensor, True)
        output = tf.nn.sigmoid(pred)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=label_tensor))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            optim = tf.train.AdamOptimizer().minimize(loss)
        return input_tensor, label_tensor, output, loss, optim

    # def worker(self, info):
    #     if len(info.strip().split("."))!=2:
    #         return None
    #     fname, suffix = info.strip().split(".")
    #     img_path = self.data_path+"Images/"+fname+".jpg"
    #     label_path = self.data_path+"train_gt_t13/"+fname+".txt"
    #     img, label_img = util.load_data(img_path, label_path)
    #     if img is None:
    #         return None
    #     return (img, label_img, fname)

    def train(self):
        input_tensor, label_tensor, output, loss, optim = self.compile()

        saver = tf.train.Saver(tf.global_variables(), max_to_keep = 1)
        restore_path = "checkpoint/"

        self.data_path = "../data/icdar2019/"

        data_list = os.listdir(self.data_path+"train_gt_t13/")
        img_fnames, label_fnames = [], []
        for name in data_list:
            if len(name.strip().split("."))!=2:
                continue
            fname, suffix = name.strip().split(".")
            img_path = self.data_path+"Images/"+fname+".jpg"
            label_path = self.data_path+"train_gt_t13/"+fname+".txt"
            if not os.path.exists(img_path):
                continue
            img_fnames.append(img_path)
            label_fnames.append(label_path)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            checkpoint = tf.train.latest_checkpoint(restore_path)

            if checkpoint:
                print("restore from: " + checkpoint)
                saver.restore(sess, checkpoint)

            dataset = tf.data.Dataset.from_tensor_slices((img_fnames, label_fnames))
            dataset = dataset.map(lambda img_f, label_f: tuple(tf.numpy_function(util.load_data, [img_f, label_f], 
                [tf.uint8, tf.uint8]))).batch(8)#.prefetch(8).shuffle(2).repeat(10)
     
            iterator = dataset.make_initializable_iterator()
            next_element = iterator.get_next()
            sess.run(iterator.initializer)

            cnt=0
            while True:
                try:
                    img, label = sess.run(next_element)
                    cnt+=1
                    _, l, o = sess.run([optim, loss, output], feed_dict={input_tensor:img, label_tensor:label})
                    print(l)
                    if cnt%3==0:
                        out_img = o[-1, :, :, :]
                        cv2.imwrite("output.jpg", out_img.astype(np.uint8)*255)
                        saver.save(sess, os.path.join(restore_path, 'model'))
                        cv2.imwrite("label.jpg", label[-1,:,:,:].astype(np.uint8)*255)
                        cv2.imwrite("input.jpg", img[-1,:,:,:].astype(np.uint8))
                except tf.errors.OutOfRangeError:
                    break



            # for epoch in range(100):
            #     pool = multiprocessing.Pool(1)
            #     batch_size = 8
            #     batch_data = []
            #     batch_label = []
            #     cnt=0
            #     for r in pool.imap_unordered(self.worker, data_list):
            #         if r is None:
            #             continue
            #         img, label_img, fname = r
            #         print(fname)
            #         batch_data.append(img)
            #         batch_label.append(label_img)
            #         if len(batch_data)>=batch_size:
            #             cnt+=1
            #             batch_data = np.array(batch_data)
            #             batch_label = np.array(batch_label).reshape([8, 256, 256, 1])
            #             _, l, o = sess.run([optim, loss, output], feed_dict={input_tensor:batch_data, label_tensor:batch_label})
            #             print(l)
            #             if cnt%3==0:
            #                 out_img = o[-1, :, :, :]
            #                 cv2.imwrite("output.jpg", out_img.astype(np.uint8)*255)
            #                 saver.save(sess, os.path.join(restore_path, 'model'))
            #                 cv2.imwrite("label.jpg", batch_label[-1,:,:,:].astype(np.uint8)*255)
            #                 cv2.imwrite("input.jpg", batch_data[-1,:,:,:].astype(np.uint8))
            #             batch_data = []
            #             batch_label = []


if __name__=="__main__":
    unet = Unet()
    unet.train()
