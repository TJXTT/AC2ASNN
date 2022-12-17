# SNN2ANN: A Fast and Memory-Efficient Training Framework for Spiking Neural Networks

The demo code of the paper: SNN2ANN: A Fast and Memory-Efficient Training Framework for Spiking Neural Networks

## SNN2ANN training with ReSU:

### VGG-13

```
python run.py --arch=VGG --time-step=4 --batch-size=512 --spike-unit=ReSU --dataset=CIFAR100 --class-nums=100 --data-path=dataset_path --pretrained=pretrained_model_path
```

### ResNet-17

```
python run.py --arch=ResNet --time-step=5 --batch-size=512 --spike-unit=ReSU --dataset=CIFAR100 --class-nums=100 --kaiming-norm=True --data-path=dataset_path --pretrained=pretrained_model_path
```

## SNN2ANN training with STSU:

### VGG-13

```
python run.py --arch=VGG --time-step=4 --batch-size=512 --spike-unit=STSU --dataset=CIFAR100 --class-nums=100 --data-path=dataset_path --pretrained=pretrained_model_path
```

### ResNet-17

```
python run.py --arch=ResNet --time-step=5 --batch-size=512 --spike-unit=STSU --dataset=CIFAR100 --class-nums=100 --kaiming-norm=True --data-path=dataset_path --pretrained=pretrained_model_path
```

## Pre-trained Model
The pre-trained models are available at
https://drive.google.com/drive/folders/1y6ZUT3WToowuCo72CVspurQrdSG4U8zi?usp=sharing

## Citation

```
@article{snn2ann,
  title={SNN2ANN: A Fast and Memory-Efficient Training Framework for Spiking Neural Networks},
  author={Jianxiong Tang and Jianhuang Lai and Xiaohua Xie and Lingxiao Yang and Wei-Shi Zheng},
  journal={arXiv preprint arXiv:2206.09449},
  year={2022}
}
```
