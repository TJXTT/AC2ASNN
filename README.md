# SNN2ANN: A Fast and Memory-Efficient Training Framework for Spiking Neural Networks

The demo code of the paper: SNN2ANN: A Fast and Memory-Efficient Training Framework for Spiking Neural Networks

## SNN2ANN training with ReSU:

### VGG-13

```
python run.py --arch=VGG --time-step=4 --batch-size=512 --spike-unit=ReSU --dataset=../datasets/data_CIFAR100 --class-num=100
```

### ResNet-17

```
python run.py --arch=ResNet --time-step=5 --batch-size=512 --spike-unit=ReSU --dataset=../datasets/data_CIFAR100 --class-num=100 --kaiming-norm=True
```

## SNN2ANN training with STSU:

### VGG-13

```
python run.py --arch=VGG --time-step=4 --batch-size=512 --spike-unit=STSU --dataset=../datasets/data_CIFAR100 --class-num=100
```

### ResNet-17

```
python run.py --arch=ResNet --time-step=5 --batch-size=512 --spike-unit=STSU --dataset=../datasets/data_CIFAR100 --class-num=100 --kaiming-norm=True
```

## Citation

```
@article{snn2ann,
  title={SNN2ANN: A Fast and Memory-Efficient Training Framework for Spiking Neural Networks},
  author={Jianxiong Tang and Jianhuang Lai and Xiaohua Xie and Lingxiao Yang and Wei-Shi Zheng},
  journal={arXiv preprint},
  year={2022}
}
```