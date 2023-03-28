Enhanced Query Featurization for Cardinality Estimation
====

Updated implementation of multi-set convolutional networks (MSCN) to include and test other featurizations [[1]](#references).

## Requirements

  * PyTorch 1.0
  * Python 3.7

## Usage

```python3 train.py --help```

Example usage:

```python3 train.py synthetic```

To use a different featurization:

```python3 train.py --feat range --queries 100000 --epochs 100 synthetic```

You can find the different featurizations in [`mscn\util.py`](https://github.com/lucaswo/queryfeaturizations/blob/master/mscn/util.py).

## References

[1] [Müller et al., Enhanced Featurization of Queries with Mixed Combinations of Predicates for ML-based Cardinality Estimation
, 2023](https://doi.org/10.48786/edbt.2023.22)

## Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{mueller2023enhanced, 
  doi = {10.48786/EDBT.2023.22}, 
  url = {https://openproceedings.org/2023/conf/edbt/paper-1.pdf}, 
  author = {M\"uller, Magnus and Woltmann, Lucas and Lehner, Wolfgang}, 
  keywords = {Database Technology}, 
  language = {en}, 
  title = {{Enhanced Featurization of Queries with Mixed Combinations of Predicates for ML-based Cardinality Estimation}}, 
  publisher = {OpenProceedings.org}, 
  year = {2023}, 
  booktitle = {Proceedings of the 26th International Conference on Extending Database Technology}, 
  location = {Ioannina, Greece},
  series = {EDBT 2023}} 
```
