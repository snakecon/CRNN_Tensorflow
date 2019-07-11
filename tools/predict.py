# coding=utf-8
from __future__ import print_function

import json

import cv2
import numpy as np
import tensorflow as tf


class CrnnInference(object):
    """
    The CRNN model.
    """
    def __init__(self, model_file, char_dict_path, ord_map_dict_path):
        self.graph = self._load_graph(model_file)

        self._char_dict = self.read_char_dict(char_dict_path)
        self._ord_map = self.read_char_dict(ord_map_dict_path)

        # Input.
        self.input_whited_image = self.graph.get_tensor_by_name('infer/input:0')
        # Output.
        self.decoder_indices = self.graph.get_tensor_by_name(
            'infer/CTCBeamSearchDecoder:0')
        self.decoder_values = self.graph.get_tensor_by_name(
            'infer/CTCBeamSearchDecoder:1')
        self.decoder_dense_shape = self.graph.get_tensor_by_name(
            'infer/CTCBeamSearchDecoder:2')

        self.sess = tf.Session(graph=self.graph)

    def inference(self, image):
        """
        Call model.
        """
        image = cv2.resize(image, (100, 32), interpolation=cv2.INTER_LINEAR)
        image = np.array(image, np.float32) / 127.5 - 1.0

        indices, values, dense_shape = self.sess.run(
            [self.decoder_indices, self.decoder_values, self.decoder_dense_shape],
            feed_dict={self.input_whited_image: [image]
        })

        # print(indices, values, dense_shape)

        preds = self.sparse_tensor_to_str(indices, values, dense_shape)[0]
        return preds

    def _load_graph(self, frozen_graph_filename):
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="infer",
                op_dict=None,
                producer_op_list=None
            )

        return graph

    def sparse_tensor_to_str(self, indices, values, dense_shape):
        """
        :param sparse_tensor: prediction or ground truth label
        :return: String value of the sparse tensor
        """
        # Translate from consecutive numbering into ord() values
        values = np.array([self._ord_map[str(tmp) + '_index'] for tmp in values])

        number_lists = np.ones(dense_shape, dtype=values.dtype)
        str_lists = []
        res = []
        for i, index in enumerate(indices):
            number_lists[index[0], index[1]] = values[i]
        for number_list in number_lists:
            # Translate from ord() values into characters
            str_lists.append([self.int_to_char(val) for val in number_list])
        for str_list in str_lists:
            # int_to_char() returns '\x00' for an input == 1, which is the default
            # value in number_lists, so we skip it when building the result
            res.append(''.join(c for c in str_list if c != '\x00'))
        return res

    def int_to_char(self, number):
        """
        convert the int index into char
        :param number: Can be passed as string representing the integer value to look up.
        :return: Character corresponding to 'number' in the char_dict
        """
        # 1 is the default value in sparse_tensor_to_str()
        # This will be skipped when building the resulting strings
        if number == 1 or number == '1':
            return '\x00'
        else:
            return self._char_dict[str(number) + '_ord']

    def read_char_dict(self, dict_path):
        """

        :param dict_path:
        :return: a dict with ord(char) as key and char as value
        """
        with open(dict_path, 'r', encoding='utf-8') as json_f:
            res = json.load(json_f)
        return res


if __name__ == '__main__':
    inference = CrnnInference(
        'export/output_graph.pb',
        'data/char_dict/char_dict_en.json',
        'data/char_dict/ord_map_en.json',
    )

    for i in [0, 1, 2, 3, 4]:
        image = cv2.imread('data/thumbnail/%d.jpg' % i, cv2.IMREAD_COLOR)
        result = inference.inference(image)
        print(result)
