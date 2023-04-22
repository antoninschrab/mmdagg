# MMDAgg package

This package implements the MMDAgg test for two-sample testing, as proposed in our paper [MMD Aggregated Two-Sample Test](https://arxiv.org/pdf/2110.15073.pdf).
The experiments of the paper can be reproduced using the [mmdagg-paper](https://github.com/antoninschrab/mmdagg-paper/) repository.
The package contains implementations both in Numpy and in Jax, we recommend using the Jax version as it runs 100 times faster after compilation (results from the notebook [demo_speed.ipynb](https://github.com/antoninschrab/mmdagg-final/blob/main/demo_speed.ipynb) in the [mmdagg-paper](https://github.com/antoninschrab/mmdagg-paper/) repository). 
The notebook also contains a demo showing how to use our MMDAgg test.
We also provide installation instructions and example code below.

| Speed in s | Numpy (CPU) | Jax (CPU) | Jax (GPU) |
| -- | -- | -- | -- |
| MMDAgg | 43.1 | 14.9 | 0.495 |

## Requirements

The requirements for the Numpy version are:
- `python 3.9`
  - `numpy`
  - `scipy`

The requirements for the Jax version are:
- `python 3.9`
  - `jax`
  - `jaxlib`

## Installation

First, we recommend creating a conda environment:
```bash
conda create --name mmdagg-env python=3.9
conda activate mmdagg-env
# can be deactivated by running:
# conda deactivate
```

We then install the required depedencies ([Jax installation instructions](https://github.com/google/jax#installation)) by running either:
- for GPU:
  ```bash
  pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
  # conda install -c conda-forge -c nvidia pip numpy scipy cuda-nvcc "jaxlib=0.4.1=*cuda*" jax
  ```
- or, for CPU:
  ```bash
  conda install -c conda-forge -c nvidia pip numpy scipy cuda-nvcc jaxlib=0.4.1 jax
  ```
  
Our `mmdagg` package can then be installed as follows:
```bash
pip install git+https://github.com/antoninschrab/mmdagg.git
```

## MMDAgg

**Two-sample testing:** Given arrays X of shape $(N_X, d)$ and Y of shape $(N_Y, d)$, our MMDAgg test `mmdagg(X, Y)` returns 0 if the samples X and Y are believed to come from the same distribution, and 1 otherwise.

**Jax compilation:** The first time the function is evaluated, Jax compiles it. 
After compilation, it can fastly be evaluated at any other X and Y of the same shape. 
If the function is given arrays with new shapes, the function is compiled again.
For details, check out the [demo_speed.ipynb](https://github.com/antoninschrab/mmdagg-final/blob/main/demo_speed.ipynb) notebook on the [mmdagg-paper](https://github.com/antoninschrab/mmdagg-paper/) repository.

```python
# import modules
>>> import jax.numpy as jnp
>>> from jax import random
>>> from mmdagg.jax import mmdagg, human_readable_dict # jax version
>>> # from mmdagg.np import mmdagg

# generate data for two-sample test
>>> key = random.PRNGKey(0)
>>> key, subkey = random.split(key)
>>> subkeys = random.split(subkey, num=2)
>>> X = random.uniform(subkeys[0], shape=(500, 10))
>>> Y = random.uniform(subkeys[1], shape=(500, 10)) + 1

# run MMDAgg test
>>> output = mmdagg(X, Y)
>>> output
Array(1, dtype=int32)
>>> output.item()
1
>>> output, dictionary = mmdagg(X, Y, return_dictionary=True)
>>> output
Array(1, dtype=int32)
>>> human_readable_dict(dictionary)
>>> dictionary
{'MMDAgg test reject': True,
 'Single test 1': {'Bandwidth': 1.0,
  'MMD': 5.788900671177544e-05,
  'MMD quantile': 0.0009193826699629426,
  'Kernel IMQ': True,
  'Reject': False,
  'p-value': 0.41079461574554443,
  'p-value threshold': 0.01699146442115307},
  ...
}
```

## MMDAggInc

For a computationally efficient version of MMDAgg which can run in linear time, check out our package `agginc` in the [agginc](https://github.com/antoninschrab/agginc) repository. 
This package implements the MMDAggInc test (together with HISCAggInc and KSDAggInc) proposed in our paper [Efficient Aggregated Kernel Tests using Incomplete U-statistics](https://arxiv.org/pdf/2206.09194.pdf) with reproducible experiments in the [agginc-paper](https://github.com/antoninschrab/agginc-paper) repository. 

## Contact

If you have any issues running our MMDAgg test, please do not hesitate to contact [Antonin Schrab](https://antoninschrab.github.io).

## Affiliations

Centre for Artificial Intelligence, Department of Computer Science, University College London

Gatsby Computational Neuroscience Unit, University College London

Inria London

## Bibtex

```
@unpublished{schrab2021mmd,
  author        = {Antonin Schrab and Ilmun Kim and M{\'e}lisande Albert and B{\'e}atrice Laurent and Benjamin Guedj and Arthur Gretton},
  title         = {{MMD} Aggregated Two-Sample Test},
  year          = {2021},
  eprint        = {2110.15073},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2110.15073},
}
```

## License

MIT License (see [LICENSE.md](LICENSE.md)).
