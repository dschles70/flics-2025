# PolyMNIST

## 1)  Prepare the data

Create the directory all the data will be stored, and go therein

```bash
mkdir data 
cd data
```

Download the data (we used instructions from [here](https://github.com/epalu/CMVAE))

```bash
curl -L -o data_PM_ICLR_2024.zip https://polybox.ethz.ch/index.php/s/DvIsHiopIoPnKXI/download
unzip data_PM_ICLR_2024.zip
```

Prepare the data for further use 

```bash
python ../prepare/prepare_polymnist.py train
```
and
```bash
python ../prepare/prepare_polymnist.py test
```

Finally, if everything went right, we recommend to remove the downloaded data to save the space
```bash
rm -rf data_PM_ICLR_2024.zip data_PM_ICLR_2024/
```
