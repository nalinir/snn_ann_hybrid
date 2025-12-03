# Directly Connected Spiking-Nonspiking Neural Network: Initial Implementation
This code implements direct spiking-nonspiking connections, as well as comparable RNN models and an initial pass at Flexible Spiking assignment in a single-layer and 2-layer setup.


## Models and Training
The models are all available [here](https://github.com/nalinir/snn_ann_hybrid/tree/main/class_based_implementation), along with train/main.py files and regularization of spiking rate, as seen in [Zenke and Vogels 2021](https://pubmed.ncbi.nlm.nih.gov/33513328/).

## Loss Landscapes
Code for loss landscape visualizations (using a shared PCA for comparability) are available [here](https://github.com/nalinir/snn_ann_hybrid/tree/main/loss_landscapes), along with a Sharpness-Aware Minimization (SAM) loss metric based on [Foret et al 2021](https://arxiv.org/pdf/2010.01412).

## Data
For comparability across models and lab-wide SNN projects for the Random Manifold Dataset (Zenke and Vogels 2021), we use a version of [this repository](https://github.com/Yixing-Wang/DarwinNeuron).

To produce SHD data, we use this folder, which is directly called in main.py [file](https://github.com/nalinir/snn_ann_hybrid/tree/main/data).


