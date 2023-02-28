# text_anomalies

This is the sourcecode for detecting anomalies in corpus data. This project tries to combine
current anomaly detection methods using variational autoencoder (VAE) and text corpus data.

This project was created in the context of the seminar series "Computing Meaning in Premodern Text Corpora" at the University of Göttingen.

## Getting Started

First of clone the repository and install the requirements:

```bash
git clone https://github.com/semohr/text_anomalies.git
# install requirements in folder
pip install .
```

Sadly I'm not allowed to share the dataset used (DOEC). But you can download it [here](http://hdl.handle.net/20.500.12024/2488) or use your own dataset. You have to do this first before you can run the code.

Once downloaded you have to extract and place inside the `data/doec_raw` folder. The folder structure should look like this:

```bash
data
└── doec_raw
    ├── doc
    ├── html
    ├── images
    ├── sgml-corpus
    ├── corpus.htm
```

Afterwards to preprocess the data run:

```bash
python scripts/preprocess.py
```

This should create a `data/doec_processed` folder with the preprocessed data in parquet format. The data can be loaded using the `DOEC` class in `text_anomalies.dataloader`. Or just using `pandas.read_parquet`.

To get started with the model you can have a look at the `notebooks/DOEC.ipynb` notebook. This notebook contains the code to train the model and run some light analysis.

## Orientation

The project is structured as follows:

```bash
├── data # folder for data
│   ├── doec_raw # folder for raw data
│   ├── doec_processed # folder for raw data
│   ├── tokenizers # folder for pretrained tokenizers
│   └── models # folder for the trained models
├── text_anomalies # folder for source code
│   ├── dataloader # code for dataloader and preprocessing
│   └── models # code for the torch modules
├── notebooks # folder for notebooks
├── scripts # folder for scripts
├── README.md # <- you are here
└── pyproject.toml
```
