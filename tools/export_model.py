#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-29 下午3:56
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/CRNN_Tensorflow
# @File    : test_shadownet.py
# @IDE: PyCharm Community Edition
"""
Use shadow net to recognize the scene text of a single image
"""
import argparse
import os
import os.path as ops

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glog as logger
import wordninja
from tensorflow.python.tools import freeze_graph

from config import global_config
from crnn_model import crnn_net
from data_provider import tf_io_pipline_fast_tools

CFG = global_config.cfg


def init_args():
    """

    :return: parsed arguments and (updated) config.cfg object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str,
                        help='Path to the image to be tested',
                        default='data/test_images/test_01.jpg')
    parser.add_argument('--weights_path', type=str,
                        help='Path to the pre-trained weights to use')
    parser.add_argument('-c', '--char_dict_path', type=str,
                        help='Directory where character dictionaries for the dataset were stored')
    parser.add_argument('-o', '--ord_map_dict_path', type=str,
                        help='Directory where ord map dictionaries for the dataset were stored')
    parser.add_argument('-v', '--visualize', type=args_str2bool, nargs='?',
                        const=True,
                        help='Whether to display images')

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def recognize(image_path, weights_path, char_dict_path, ord_map_dict_path,
              is_vis, is_english=True):
    """

    :param image_path:
    :param weights_path:
    :param char_dict_path:
    :param ord_map_dict_path:
    :param is_vis:
    :return:
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    new_heigth = 32
    scale_rate = new_heigth / image.shape[0]
    new_width = int(scale_rate * image.shape[1])
    new_width = new_width if new_width > CFG.ARCH.INPUT_SIZE[0] else \
    CFG.ARCH.INPUT_SIZE[0]
    # TODO: Fix it,  force 100.
    new_width = 100

    image = cv2.resize(image, (new_width, new_heigth),
                       interpolation=cv2.INTER_LINEAR)
    image_vis = image
    image = np.array(image, np.float32) / 127.5 - 1.0

    print(new_width, new_heigth)

    inputdata = tf.placeholder(
        dtype=tf.float32,
        shape=[1, new_heigth, new_width, CFG.ARCH.INPUT_CHANNELS],
        name='input'
    )

    codec = tf_io_pipline_fast_tools.CrnnFeatureReader(
        char_dict_path=char_dict_path,
        ord_map_dict_path=ord_map_dict_path
    )

    net = crnn_net.ShadowNet(
        phase='test',
        hidden_nums=CFG.ARCH.HIDDEN_UNITS,
        layers_nums=CFG.ARCH.HIDDEN_LAYERS,
        num_classes=CFG.ARCH.NUM_CLASSES
    )

    inference_ret = net.inference(
        inputdata=inputdata,
        name='shadow_net',
        reuse=False
    )

    decodes, _ = tf.nn.ctc_beam_search_decoder(
        inputs=inference_ret,
        sequence_length=int(new_width / 4) * np.ones(1),
        merge_repeated=True,
        beam_width=10
    )
    decode = decodes[0]

    print(decode)
    # config tf saver
    saver = tf.train.Saver()

    # config tf session
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TEST.TF_ALLOW_GROWTH

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        preds = sess.run(decode, feed_dict={inputdata: [image]})

        print(preds)

        preds = codec.sparse_tensor_to_str(preds)[0]
        if is_english:
            preds = ' '.join(wordninja.split(preds))

        # return preds_evaluated
        input_graph_name = "input_graph.pb"
        output_graph_name = "output_graph.pb"
        export_dir = 'export'
        tf.train.write_graph(sess.graph, export_dir, input_graph_name)
        tf.logging.info("Write graph at %s." % os.path.join(export_dir,
                                                            input_graph_name))

        export_graph = tf.Graph()
        with export_graph.as_default():
            freeze_graph.freeze_graph(input_graph=os.path.join(export_dir,
                                                               input_graph_name),
                                      input_saver="",
                                      input_binary=False,
                                      input_checkpoint=weights_path,
                                      output_node_names='CTCBeamSearchDecoder',
                                      restore_op_name="",
                                      filename_tensor_name="",
                                      output_graph=os.path.join(export_dir,
                                                                output_graph_name),
                                      clear_devices=True,
                                      initializer_nodes=None,
                                      variable_names_blacklist="")

        tf.logging.info("Export model at %s." % os.path.join(export_dir,
                                                             output_graph_name))


        logger.info('Predict image {:s} result: {:s}'.format(
            ops.split(image_path)[1], preds)
        )

        if is_vis:
            plt.figure('CRNN Model Demo')
            plt.imshow(image_vis[:, :, (2, 1, 0)])
            plt.show()

    sess.close()

    return


if __name__ == '__main__':
    """

    """
    # init images
    args = init_args()

    # detect images
    recognize(
        image_path='data/thumbnail/4.jpg',
        weights_path='checkpoints/shadownet.ckpt',
        char_dict_path='data/char_dict/char_dict_en.json',
        ord_map_dict_path='data/char_dict/ord_map_en.json',
        is_vis=True
    )
