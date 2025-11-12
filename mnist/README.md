# MNIST

In order to learn a pair of models according to the proposed approach (see the paper, section IV-A), run e.g.
```python
python main.py --id 0 --stepsize 1e-4 --niterations 1000000 --alpha 0.1
```
The crucial parameter here is `id`, which is a call identifier (other parameters are rather self-explaining, see the source for more details). The script `main.py` creates (if necessary), three subdirectories -- `./logs`, `./models` and `./images` -- and stores the learning results there under names containing `id`. Hence, later, the results of particular calls can be addressed. Such a learning takes about 15 hours on a single NVIDIA H100 GPU. One can watch the learning using the notebook `plot.ipynb` (losses, some images, etc.), which reads the logs and the images and plots them.

Repeat the learning for the series of models varying `id` from 0 to 9 and `alpha` from 0.1 to 1.0 correspondingly.

In order to learn the baseline model, invoke
```python
python main_baseline.py --id baseline --stepsize 1e-4 --niterations 1000000
```
which just learns the baseline HVAE on the whole dataset.

In order to generate an image like in Fig. 1 (top) in the paper, run `python generate.py`.
___

In order to compute FID scores, quite many intermediate steps should be performed -- generating images from different models, computing FID scores between different subsets, etc. So, we packed everything in a shell script `fid_script.sh`, just call it. For generating images from learned models, `generate_data.py` is used; for computing FID scores, we used the code from [here](https://github.com/mseitzer/pytorch-fid). If everything goes smoothly, the script takes about 3 hours for computing all the necessary numbers. They are stored under `./fids/out*.txt`, which can be then plotted using `plot_fids.ipynb`, producing a figure like Fig. 1 (bottom) in the paper.
___

The training scripts `main*.py` produce the folder `./data` where the original dataset is downloaded. The script `fid_script.sh` stores generated images under `./generated_data`. Both can be safely removed after the work is done.
