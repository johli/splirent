# SPLIRENT - Splicing Regression Net
This repository contains the code for training and running SPLIRENT, a deep neural network that can predict alternative 5' splice donor usage given the DNA sequence as input, with cell type-specific predictions for HEK, HELA, MCF7 and CHO.

SPLIRENT was trained on the randomized splicing MPRA described in [Rosenberg *et al*, Cell 2015](https://doi.org/10.1016/j.cell.2015.09.054). The MPRA was replicated in cell lines HELA, MCF7 and CHO and merged with the original HEK-measured dataset.

This readme contain links to IPython Notebooks containing analyses and model evaluations. There is also a link to the repository containing all of the processed data used in the notebooks.

Contact *jlinder2 (at) cs.washington.edu* for any questions about the model or data.

### Installation
SPLIRENT can be installed by cloning or forking the [github repository](https://github.com/johli/splirent.git):
```sh
git clone https://github.com/johli/splirent.git
cd splirent
python setup.py install
```

#### SPLIRENT requires the following packages to be installed
- Python >= 3.6
- Tensorflow >= 1.13.1
- Keras >= 2.2.4
- Scipy >= 1.2.1
- Numpy >= 1.16.2
- Isolearn >= 0.2.0 ([github](https://github.com/johli/isolearn.git))
- [Optional] Pandas >= 0.24.2
- [Optional] Matplotlib >= 3.1.1

### Usage
SPLIRENT is built as a Keras Model, and as such can be easily executed using simple Keras function calls.
See the example usage notebooks below for a tutorial on how to use the model.

## Data Availability
The processed 5' Alternative Splicing MPRA data is stored in the repository below. The dataset is stored as a .tar.gz archive at the following path:

**alt_5ss/unprocessed_data/alt_5ss_multi_cell_line_data.tar.gz**
> The base version of the dataset, containing a .csv file of splice variant sequences and a .mat file of corresponding cell type-specific RNA-Seq read counts supporting 5' splicing at each nucleotide position.

[Processed Data Repository](https://drive.google.com/drive/folders/1M0FLhOOw_lD8sAAmsuRUWq5ZEyrNVv2n?usp=sharing)<br/>

## Analysis
The following collection of IPython Notebooks contains model analyses (cell type-specific prediction accuracies, differential splicing, etc.).

### Random MPRA Logistic Regression (Hexamer) Notebooks
Evaluation of the Linear Logistic Hexamer Regression trained on the Random MPRA library.

[Notebook 1a: Logistic Regression Training](https://nbviewer.jupyter.org/github/johli/splirent/blob/master/logistic_regression/splirent_5ss_logistic_regression_multicell.ipynb)<br/>
[Notebook 1b: Logistic Regression Evaluation](https://nbviewer.jupyter.org/github/johli/splirent/blob/master/logistic_regression/splirent_5ss_logistic_regression_multicell_eval.ipynb)<br/>
[Notebook 2a: Logistic Regression Training (both random regions)](https://nbviewer.jupyter.org/github/johli/splirent/blob/master/logistic_regression/splirent_5ss_logistic_regression_both_regions_multicell.ipynb)<br/>
[Notebook 2b: Logistic Regression Evaluation (both random regions)](https://nbviewer.jupyter.org/github/johli/splirent/blob/master/logistic_regression/splirent_5ss_logistic_regression_both_regions_multicell_eval.ipynb)<br/>

### Random MPRA Neural Network (CNN) Notebooks
Evaluation of the SPLIRENT CNN trained on the Random MPRA library.

[Notebook 3a: Neural Network Evaluation (isoforms)](https://nbviewer.jupyter.org/github/johli/splirent/blob/master/misc/evaluate_a5ss_splirent_model_only_random_regions_sgd.ipynb)<br/>
[Notebook 3b: Neural Network Evaluation (nt-resolution)](https://nbviewer.jupyter.org/github/johli/splirent/blob/master/misc/evaluate_a5ss_splirent_model_only_random_regions_cuts_sgd.ipynb)<br/>
