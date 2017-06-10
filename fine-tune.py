import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet
from common import data, fit, modelzoo
import mxnet as mx

import os, urllib
def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists('model/'+filename):
        urllib.urlretrieve(url, 'model/'+ filename)

def get_model(prefix, epoch):
    download(prefix+'-symbol.json')
    download(prefix+'-%04d.params' % (epoch,))

def get_fine_tune_model(symbol, arg_params, num_classes, layer_name):
    """
    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = sym.get_internals()
    net = all_layers[layer_name+'_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc') #, lr_mult=10)
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc' not in k})
    return (net, new_args)


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="fine-tune a dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train = fit.add_fit_args(parser)
    data.add_data_args(parser)
    aug = data.add_data_aug_args(parser)
    parser.add_argument('--pretrained-model', type=str,
                        help='the pre-trained model')
    parser.add_argument('--layer-before-fullc', type=str, default='flatten0',
                        help='the name of the layer before the last fullc layer')
    # use less augmentations for fine-tune
    data.set_data_aug_level(parser, 1)
    # use a small learning rate and less regularizations
    # when training comes to 10th and 20th epoch
	# see http://mxnet.io/how_to/finetune.html and Mu's thesis
    # http://www.cs.cmu.edu/~muli/file/mu-thesis.pdf 
    parser.set_defaults(image_shape='3,320,320', num_epochs=30,
                        lr=.01, lr_step_epochs='10,20', wd=0, mom=0)

    args = parser.parse_args()

    # load pretrained model
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
	# get the pretrained resnet 152 from official MXNet model zoo
	# 1k imagenet pretrained
    #get_model('http://data.mxnet.io/models/imagenet/resnet/152-layers/resnet-152', 0)
	# 11k imagenet resnet 152 has stronger classification power
     get_model('http://data.mxnet.io/models/imagenet-11k/resnet-152/resnet-152', 0)
    prefix = 'model/resnet-152'
    epoch = 0
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    # remove the last fullc layer
    (new_sym, new_args) = get_fine_tune_model(
        sym, arg_params, args.num_classes, args.layer_before_fullc)

    
    # train
    fit.fit(args        = args,
            network     = new_sym,
            data_loader = data.get_rec_iter,
            arg_params  = new_args,
            aux_params  = aux_params)
