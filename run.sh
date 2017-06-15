export MXNET_CPU_WORKER_NTHREADS=48
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python fine-tune.py --pretrained-model model/resnet-152 \
    --load-epoch 0 --gpus 0,1,2,3 \
	--model-prefix model/iNat-resnet-152 \
	--data-nthreads 48 \
    --batch-size 48 --num-classes 5089 --num-examples 579184
