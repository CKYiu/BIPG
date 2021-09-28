# BIPG

Code repository for our paper “Boundary Information Progressive Guidance
Network for Salient Object Detection”, accepted by IEEE Transactions on Multimedia.

## Prerequisites
- [Pytorch 1.7](http://pytorch.org/)


## Code
### test
-Please download the ResNet-50 ([pretrained backbone](https://download.pytorch.org/models/resnet50-19c8e357.pth)) into `PyTorch Pretrained\ResNet` folder

-Please download the pretrained model for BIPG into `model` folder, which can be found at ([Baidu Netdisk](https://pan.baidu.com/s/1KHbTmHm_0VY8eCc-R6SDHg ) (code:sysu) or [Google Drive](https://drive.google.com/file/d/1cJzwzeXJPB5G2IOccbX5rIRu56zykL5C/view?usp=sharing))


## Quantitative Performance
|       | F_measure (max) | F_measure (avg) | weight F_measure | S_measure | E_measure | MAE|
| :------:| :--------: | :--------: |:--------: | :--------: | :--------: | :--------:| 
| ECSSD | 0.953 | 0.938 |0.926 | 0.929 | 0.958 | 0.029|
| PASCAL-S| 0.880 | 0.853 |0.833 | 0.861 | 0.907 | 0.059|
| HKU-IS| 0.943 | 0.926 |0.916 | 0.924 | 0.964 | 0.025|
| DUT-OMRON | 0.824 | 0.788 |0.772 | 0.845 | 0.888 | 0.051|
| DUTS-test| 0.900 | 0.871 |0.861 | 0.897 | 0.936 | 0.033|

## Saliency maps
The saliency maps produced by our network can be found at [Baidu Netdisk](https://pan.baidu.com/s/1lwk7xPFshcWfhaTj-ZfLVA) (code:bipg) or [Google Drive](https://drive.google.com/file/d/1RFUL_NrHX3_NR0GLj2sTi0R3nNTIyL1F/view?usp=sharing).


