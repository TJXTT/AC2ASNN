# $\text{AC}^2\text{AS}$: Activation Consistency Coupled ANN-SNN Framework for Fast and Memory-Efficient SNN Training

The demo code of the paper: $\text{AC}^2\text{AS}$: Activation Consistency Coupled ANN-SNN Framework for Fast and Memory-Efficient SNN Training

## Training with ReSU:

### VGG-13

```
python run.py --arch=VGG --time-step=4 --batch-size=512 --spike-unit=ReSU --dataset=CIFAR100 --class-nums=100 --data-path=dataset_path --pretrained=pretrained_model_path
```

### ResNet-17

```
python run.py --arch=ResNet --time-step=5 --batch-size=512 --spike-unit=ReSU --dataset=CIFAR100 --class-nums=100 --kaiming-norm=True --data-path=dataset_path --pretrained=pretrained_model_path
```

## Training with STSU:

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
@article{ac2asnn,
  title={AC2AS: Activation Consistency Coupled ANN-SNN Framework for Fast and Memory-Efficient SNN Training},
  author={Jianxiong Tang and Jianhuang Lai and Xiaohua Xie and Lingxiao Yang and Wei-Shi Zheng},
  journal={Pattern Recognition},
  year={2023}
}
```
