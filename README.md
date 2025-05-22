
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
    ├── xlsa17/
```
then run some scripts to prepare data.

```
python scripts/get_allclasses.py
python scripts/make_att_npy.py

get_attr_awa2.py
get_attr_cub.py
get_attr_sun.py
```

## Train

```
python train.py --config configs/templates/pixel.yaml --ds awa2/cub/sun --nbit 24/48/64/128 --R 5000
```


