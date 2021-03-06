# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi he, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths_fr
from model.test import test_net
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys

import tensorflow as tf
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.P4 import P4
from utils.logger import setup_logger


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--model', dest='model',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)
    parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default='', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152, mobile',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # print('Called with args:')
    # print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # if has model, get the name from it
    # if does not, then just use the initialization weights
    if args.model:
        filename = os.path.splitext(os.path.basename(args.model))[0]
        print('1....................................................',filename)
        print(args.model)
    else:
        filename = os.path.splitext(os.path.basename(args.weight))[0]
        print('2....................................................',filename)

    tag = args.tag
    tag = tag if tag else 'default'
    filename = tag + '/' + filename
    #filename ??? default/res101_faster_rcnn_iter_70000

    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)

    output_dir = get_output_dir(imdb, filename)
    logger = setup_logger("faster-rcnn", save_dir=output_dir, filename="log_test.txt")
    logger.info('Called with args:')
    logger.info(args)
    logger.info('Using config:\n{}' .format(pprint.pformat(cfg)))
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    if cfg.P4:
        # load network
        if args.net == 'res50':
            net = P4(num_layers=50)
        elif args.net == 'res101':
            net = P4(num_layers=101)
        else:
            raise NotImplementedError
    else:
        # load network
        if args.net == 'vgg16':
            net = vgg16()
        elif args.net == 'res50':
            net = resnetv1(num_layers=50)
        elif args.net == 'res101':
            net = resnetv1(num_layers=101)
        else:
            raise NotImplementedError

    # load model
    net.create_architecture("TEST", imdb.num_classes, tag='default',
                            anchor_sizes=cfg.ANCHOR_SIZES,
                            anchor_strides=cfg.ANCHOR_STRIDES,
                            anchor_ratios=cfg.ANCHOR_RATIOS)

    if args.model:
        logger.info(('Loading model check point from {:s}').format(args.model))
        print(        logger.info(('Loading model check point from {:s}').format(args.model)))
        saver = tf.train.Saver()#????????????????????????tensor???????????????????????????
        saver.restore(sess, args.model)
        logger.info('Loaded.')
    else:
        logger.info(('Loading initial weights from {:s}').format(args.weight))
        sess.run(tf.global_variables_initializer())
        logger.info('Loaded.')

    test_net(sess, net, imdb, filename, max_per_image=args.max_per_image)

    sess.close()

