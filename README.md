# iNaturalist
MXNet fine-tune baseline script (resnet 152 layers) for iNaturalist Challenge at FGVC 2017, public LB score 0.117 from a single 21st epoch submission without ensemble.

## How to use

### Install MXNet 

Run `pip install mxnet-cu80` after installing CUDA driver or go to <https://github.com/dmlc/mxnet/> for the latest version from Github.

Windows users? no CUDA 8.0? no GPU? Please run `pip search mxnet` and find the good package for your platform.

### Generate lists

After downloading and unzipping the train and test set in to `data`, along with the necessary `.json` annotation files, run `python mx_list.py` under `data` and generate `train.lst` `val.lst` `test.lst`

### Generate rec files

A good way to speed up training is maximizing the IO by using `.rec` format, which also provides convenience of data augmentation. In the `data/` directory, `gen_rec.sh` can generate `train.rec` and `val.rec` for the train and validate datasets, and `im2rec.py` can be obtained from MXNet repo <https://github.com/dmlc/mxnet/tree/master/tools> . One can adjust `--quality 95` parameter to lower quality for saving disk space, but it may take risk of loosing training precision.

### Train

Run `sh run.sh` which looks like (a 4 GTX 1080 machine for example):

```
python fine-tune.py --pretrained-model model/resnet-152 \
    --load-epoch 0 --gpus 0,1,2,3 \
    --model-prefix model/iNat-resnet-152 \
	--data-nthreads 48 \
    --batch-size 48 --num-classes 5089 --num-examples 579184
```

please adjust `--gpus` and `--batch-size` according to the machine configuration. A sample calculation: `batch-size = 12` can use 8 GB memory on a GTX 1080, so `--batch-size 48` is good for a 4-GPU machine.

Please have internet connection for the first time run because needs to download the pretrained model from <http://data.mxnet.io/models/imagenet-11k/resnet-152/>. If the machine has no internet connection, please download the corresponding model files from other machines, and ship to `model/` directory.

### Generate submission file

After a long run of some epochs, e.g. 30 epochs, we can select some epochs for the submission file. Run `sub.py` which two parameters : `num of epoch` and `gpu id` like:

```
python sub.py 21 0
```

selects the 21st epoch and infer on GPU `#0`. One can merge multiple epoch results on different GPUs and ensemble for a good submission file.

## How 'fine-tune' works

Fine-tune method starts with loading a pretrained ResNet 152 layers (Imagenet 11k classes) from MXNet model zoo, where the model has gained some prediction power, and applies the new data by learning from provided data. 

The key technique is from `lr_step_epochs` where we assign a small learning rate and less regularizations when approach to certain epochs. In this example, we give `lr_step_epochs='10,20'` which means the learning rate changes slower when approach to 10th and 20th epoch, so the fine-tune procedure can converge the network and learn from the provided new samples. A similar thought is applied to the data augmentations where fine tune is given less augmentation. This technique is described in Mu's thesis <http://www.cs.cmu.edu/~muli/file/mu-thesis.pdf> 

This pipeline is not limited to ResNet-152 pretrained model. Please experiment the fine tune method with other models, like ResNet 101, Inception, from MXNet's model zoo <http://data.mxnet.io/models/> by following this tutorial <http://mxnet.io/how_to/finetune.html> and this sample code <https://github.com/dmlc/mxnet/blob/master/example/image-classification/fine-tune.py> . Please feel free submit issues and/or pull requests and/or discuss on the Kaggle forum if have better results.

## Reference

* MXNet's model zoo <http://data.mxnet.io/models/>
* MXNet fine tune <http://mxnet.io/how_to/finetune.html> <https://github.com/dmlc/mxnet/blob/master/example/image-classification/fine-tune.py>
* Mu Li's thesis <http://www.cs.cmu.edu/~muli/file/mu-thesis.pdf> 
* iNaturalist Challenge at FGVC 2017 <https://www.kaggle.com/c/inaturalist-challenge-at-fgvc-2017/>