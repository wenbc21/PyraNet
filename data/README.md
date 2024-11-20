# Dataset

## MPII
The official website of MPII is
```
http://human-pose.mpi-inf.mpg.de/
```
where you can download the MPII dataset.


## LSP
You can download the dataset at official website
```
http://sam.johnson.io/research/lspet.html
```
As some links of LSP dataset are invalid, you may download the dataset at my google drive
```
https://drive.google.com/file/d/10SYl-IFd47yY_XiFh_q44xnsnUYM4S4P/view?usp=sharing
```

You may also download LSP by some internet links:
```
https://github.com/axelcarlier/lsp, https://www.kaggle.com/datasets/dkrivosic/leeds-sports-pose-lsp
```
but the format might be different.


## Formatting
You should organize the data directory like this.
```
data
├── lsp
│   ├── images
│   │   ├── im0001.jpg
│   │   ├── im0002.jpg
│   │   ├── ...
│   ├── LEEDS_annotations.json
│   ├── ...
├── mpii
│   ├── images
│   │   ├── 000001163.jpg
│   │   ├── 000003072.jpg
│   │   ├── ...
│   ├── annot
│   │   ├── train.h5
│   │   ├── val.h5
│   │   ├── test.h5
```