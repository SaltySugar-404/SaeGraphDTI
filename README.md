# SaeGraphDTI: drug-target interaction prediction based on sequence attribute extraction and graph neural network
Accurately identifying drug-target interactions (DTI) can greatly shorten the drug development cycle and reduce the cost of drug development. In current deep learning-based DTI prediction models, the extraction of drug and target features is a key step to improve model performance. At the same time, drugs and targets form a complex relational network, and leveraging existing network topological relationships can obtain more comprehensive feature representations.
		
We propose a DTI prediction model based on sequence feature extraction and graph neural networks, named SaeGraphDTI. First, sequence feature extractors are applied to extract relevant properties of drug and target sequences. Then, based on similarity relationships, the existing relational network is supplemented, and the graph encoder updates node information based on this network. Finally, the graph decoder calculates the probability of edge existence to predict DTI. Our model was compared with other state-of-the-art methods on four public datasets and achieved the best results in most key metrics.
		
These results demonstrate the excellent capability of this model in predicting potential DTI, providing a valuable tool for drug development.

## Requirements
- numpy
- pandas
- torch >=2.1.2
- torch-geometric >=2.4.0
- selfies >= 2.1.1

## Datasets
We provide four datasets in the `data` folder: E, GPCR, IC, and DAVIS. You can add new datasets by placing a `dataset.csv` under `data/new_dataset_id`.

## Configs
You can set the parameters for the current dataset by modifying the `.configs` of the `DatasetWithConfigs` class in the `configs.py`. Set `load_best_configs` to `False` to use the current configuration for training; otherwise, the best configuration will be used.

## Run
Modify the `dataset_id` in `configs.py` and run `main.py` to experiment with the current dataset.