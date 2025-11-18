# PolyMNIST

Here we describe the pipeline to reproduce the results from section IV-C. In contrast to the other experiments, the procedure here is slightly more complex, since we have to train different models operating on different subsets of random variables, etc. So below are step-by-step instructions to reproduce results from Fig. 3 in the paper.

## 1) Prepare the data

Create the directory where all the data will be stored, and go therein.

```bash
mkdir data 
cd data
```

Download the data (we used instructions from [here](https://github.com/epalu/CMVAE)).

```bash
curl -L -o data_PM_ICLR_2024.zip https://polybox.ethz.ch/index.php/s/DvIsHiopIoPnKXI/download
unzip data_PM_ICLR_2024.zip
```

Prepare the data for further use. 

```bash
python ../prepare/prepare_polymnist.py train
```
and
```bash
python ../prepare/prepare_polymnist.py test
```

Finally, if everything went right, we recommend removing the downloaded data to save space.
```bash
rm -rf data_PM_ICLR_2024.zip data_PM_ICLR_2024/
```

## 2) Train the baseline model

To remember, it is the baseline classifier $p(c|x)$, which is implemented as a simple Feed-Forward Network.

1. Go to `./baseline`.
2. Run ```python main.py --id 0 --niterations 1000000``` to train (takes about 1.5 hours).
3. Do inference by calling ```python inference.py --id 0```. Note that the inference results (classification accuracies per style as well as the average one) are printed out into the standard output. Store this output somewhere (e.g. redirect stdout to a file) to use later.

## 3) Train the "lightweight" model

To remember, it is a model which serves as a source of synthetic pairs $(c,s)$ for the "main" model (see later). In essence, it is a conditional VAE for $p(s|c)$, where $p(c)$ (digits) is assumed uniform. The model is trained on the binarized MNIST dataset using Symmetric Learning.

1. Go to `./cs-generator`
2. Call ```python main.py --id 0 --stepsize 1e-4 --niterations 1000000 --nz 16``` to train (takes about 5 hours).
3. Call ```python export.py --id 0 --nz 16```. Here, we use the scripting functionality of PyTorch to prepare a ready-to-deploy model for convenience. Such a model can be just loaded and used as a ``black box'' data generator.

## 4) Train the final model

1. Go to `./main_model`
2. Run `python main.py --id 0 --nz 16 --ny 256 --niterations 1000000`
and `python main.py --id 1 --nz 16 --ny 256 --niterations 1000000 --generator_path ../cs-generator/export/model_0.pt` for learning with real data and synthetic data respectively (takes about 21.5 and 36 hours respectively).
3. Compute classification accuracies per style by `python inference.py --id 0 --nz 16` and `python inference.py --id 1 --nz 16` for the obtained models. Similarly to the baseline model learned in 2), the inference results are printed out into the standard output. Store them somewhere to use later.
4. The images like the ones presented in Fig. 3 (top and middle) can be produced by invoking `python generate.py --id 1 --nz 16 --ny 256 --generator_path ../cs-generator/export/model_0.pt`. They are stored under `./images/generated_1.png` and `./images/original.png` respectively.
5. In order to plot the computed accuracies, put the stored accuracy values into the corresponding place of `plot_acc.ipynb` (see the notebook for details).
