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

## 2) Train the baseline model

1. Go to `./baseline`
2. Run ```python main.py --id 0 --niterations 1000000``` to train
3. Do inference by calling ```python inference.py --id 0```. Note that the inference results (classification accuracies per style as well as the average one) are printed out into the standard output. Store this output somewhere (e.g. redirect stdout to a file) to use later.

## 3) The "lightweight" model 

To remember, it is a model which serves as ...
1. Go to `./cs-generator`
2. Call ```python main.py --id 0 --stepsize 1e-4 --niterations 1000000 --nz 16``` to train
3. Call ```python export.py --id 0 --nz 16```