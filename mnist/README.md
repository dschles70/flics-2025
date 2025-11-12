# MNIST

In order to learn a pair of models according to the proposed approach (see the paper, section IV-A), run e.g.
```python
python main.py --id 0 --stepsize 1e-4 --niterations 1000000 --alpha 0.1
```
The crucial parameter here is `id`, which is a call identifier (other parameters are rather self-speaking, see the source for more details). The script `main.py` creates (if necessary), three subdirectories -- `./logs`, `./models` and `./images` -- and stores the learning results there under names containing `id`. Hence, later, the results of particular calls can be addressed. Such a learning takes about 15 hours on a single NVIDIA H100 GPU. One can watch the learning using the notebook `plot.ipynb` (losses, some images, etc.), which reads the logs and the images and plots them.

Repeat the learning for the series of models varying `id` from 0 to 9 and `alpha` from 0.1 to 1.0 correspondingly.

In order to learn the baseline model, invoke
```python
python main_baseline.py --id baseline --stepsize 1e-4 --niterations 1000000
```
which just learns the baseline HVAE on the whole dataset.

In order to generate image like in Fig. 1 (top) in the paper, invoke `python3 ./generate.py`

___

In order to compute FID scores, we should first prepare the image sets to compare. Use `python ./generate_data_orig.py` to prepare two original image sets -- for digits 0 to 4 and 5 to 9 respectively. They are stored under `./generated_data/origin1/`. For the whole dataset we just joined these manually as follows:
```bash
> cd ./generated_data/
> mkdir origin
> cd origin
> cp ../origin1/first/*.* ./
> cp ../origin1/second/*.* ./
```
