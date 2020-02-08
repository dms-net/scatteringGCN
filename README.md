# scatteringGCN

## Reference:
**Yimeng Min, Frederik Wenkel and Guy Wolf**\
*Scattering GCN: overcoming oversmoothness in graph convolutional networks*

## To run the Code
(default:Cora)\
for example run:\
python train.py --dataset cora --hid1 20 --hid2 20 --l1 0.005 --epochs 200 --sct_inx1 3 --dropout 0.9 --smoo 0.1

## Details of the parameters
1. hid1
2. hid2
3. hid3
4. l1
5. epochs
6. sct_inx1
7. sct_inx2
8. dropout
9. smoo

## Training curve

## requirement:
pytorch\
cuda\
scipy: for the sparse matrix operation 

