# scatteringGCN


### To run the Code
(default:Cora)
run:
python train.py --dataset cora --hid1 20 --hid2 20 --l1 0.005 --epochs 200 --sct_inx1 3 --dropout 0.9 --smoo 0.1




requirement:
pytorch
cuda

Reference:
Yimeng Min, Frederik Wenkel and Guy Wolf
Scattering GCN: overcoming oversmoothness in graph convolutional networks
