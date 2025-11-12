# MNIST

In order to learn a pair of models according to the proposed approach (see the paper, section IV-A), run e.g.
```python
python main.py --id 0 --stepsize 1e-4 --niterations 1000000 --alpha 0.1
```
The crucial parameter here is ``id``, which is a call identifier (other parameters are rather self-speaking, see the source for more details). The script ``main.py`` creates (if necessary), three subdirectories -- ``./logs``, ``./models`` and ``./images`` -- and stores the learning results there under names containing ``id``. Hence, later, the results of particular calls can be addressed. Such a learning takes about 15 hours on a single NVIDIA H100 GPU. One can watch the learning using the notebook ``plot.ipynb`` (losses, some images, etc.), which reads the logs and the images and plots them.

In order to learn the baseline model, invoke
```python
python main_baseline.py --id baseline --stepsize 1e-4 --niterations 1000000
```
which just learns the baseline HVAE on the whole dataset.
