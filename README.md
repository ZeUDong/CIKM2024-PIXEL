
# CIKM2024-PIXEL

Our implementation is built upon the open-source [Fast Image Retrieval (FIRe) framework](https://github.com/CISiPLab/cisip-FIRe), with modifications made to the data loader to accommodate zero-shot hashing tasks. And other modifications to make our improvement, such as mutli-modal data input, loss design, etc.







## Installation
Please head up to [Get Started Docs](https://fast-image-retrieval.readthedocs.io/en/latest/get_started.html) for guides on setup conda environment and installation.

## Datasets
|Dataset|Name in framework|
|---|---|
|Animals with Attributes 2 |awa2|
|CUB200-2011|cub|
|SUN Attribute|sun|

## Data Prepare
Download the awa2, cub, sun official data and put into data/, and download [xlas17](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip) put there, too. For zero-shot task stanard split.

```plaintext
├── data/
    ├── AwA2/
		├── Animals_with_Attributes2/
		├── AwA2-filenames.txt
		├── ...
    ├── CUB_200_2011/
		├── attributes/
		├── images/
		├── classes.txt
		├── ...
    ├── SUN/
		├── images/
		├── SUNAttributeDB/
		├── sun_att.npy
		├── ...
    ├── xlsa17/
```
then run some scripts to prepare data.

```shell
python scripts/get_allclasses.py

python scripts/make_att_npy.py

python get_attr_awa2.py
python get_attr_cub.py
python get_attr_sun.py
```

## Train & Eval

```shell
python train.py --config configs/templates/pixel.yaml --ds awa2 --nbit 64 --R 5000

python train.py --config configs/templates/pixel.yaml --ds cub --nbit 64 --R 5000

python train.py --config configs/templates/pixel.yaml --ds sun --nbit 64 --R 5000
```



## Reference
If you find this code useful in your research, please consider cite this project.

```bibtex
@inproceedings{dong2024pixel,
  title={PIXEL: Prompt-based Zero-shot Hashing via Visual and Textual Semantic Alignment},
  author={Dong, Zeyu and Long, Qingqing and Zhou, Yihang and Wang, Pengfei and Zhu, Zhihong and Luo, Xiao and Wang, Yidong and Wang, Pengyang and Zhou, Yuanchun},
  booktitle={Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
  pages={487--496},
  year={2024}
}
```