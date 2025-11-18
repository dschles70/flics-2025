# FMNIST

The code structure is very similar to the MNIST experiments. Hence, here we just provide commands necessary to reproduce the experiment, in particular Fig. 2 from section IV-B. Note that here we compare different learning methods of the same model. To simplify the usage, we wrote a dedicated training script for each method (they are indeed very similar to each other). Also, note that the parameter `bs` defines the number of training examples _per digit_, i.e. the real training set size is 10 times more.

To train all the models necessary to reproduce the results, run:

```bash
python main_discr.py --id 0 --stepsize 1e-4 --niterations 200000 --bs 10
python main_discr.py --id 1 --stepsize 1e-4 --niterations 200000 --bs 50
python main_discr.py --id 2 --stepsize 1e-4 --niterations 200000 --bs 100
python main_discr.py --id 3 --stepsize 1e-4 --niterations 200000 --bs 500

python main_dist.py --id 10 --stepsize 1e-4 --niterations 200000 --bs 10
python main_dist.py --id 11 --stepsize 1e-4 --niterations 200000 --bs 50
python main_dist.py --id 12 --stepsize 1e-4 --niterations 200000 --bs 100
python main_dist.py --id 13 --stepsize 1e-4 --niterations 200000 --bs 500

python main_symm.py --id 20 --stepsize 1e-4 --niterations 200000 --bs 10
python main_symm.py --id 21 --stepsize 1e-4 --niterations 200000 --bs 50
python main_symm.py --id 22 --stepsize 1e-4 --niterations 200000 --bs 100
python main_symm.py --id 23 --stepsize 1e-4 --niterations 200000 --bs 500

python main_coop.py --id 30 --stepsize 1e-4 --niterations 200000 --bs 10
python main_coop.py --id 31 --stepsize 1e-4 --niterations 200000 --bs 50
python main_coop.py --id 32 --stepsize 1e-4 --niterations 200000 --bs 100
python main_coop.py --id 33 --stepsize 1e-4 --niterations 200000 --bs 500

python main_baseline.py --id 40 --stepsize 1e-4 --niterations 200000
```
You can compose the above command into a shell script, run in parallel, whatever, depending on your computational environment. In doing so, note that some of the processes can take quite a long time, since in fact, four models are learned on a single GPU (our implementation is far from being efficient yet). For example, the call with `--id 33`, which is the proposed approach with the largest training set, takes about 24 hours on a single NVIDIA H100 GPU.

The training can be watched by `plot.ipynb`. After all models are learned, the accuracy graphs can be visualized by `plot_acc.ipynb`.
