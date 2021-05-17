This is the repository of our PIC project on news recommendation federated learning.

NRMS model is implemented federatedly via Pysyft.

## Requirements
python==3.6

Pysyft==0.2.x

torch==1.4.0

tensorboard==2.5.0

For installation of Pysyft_0.2.x, please check the Pysyft website: [Pysyft_0.2.x](https://github.com/OpenMined/PySyft/tree/syft_0.2.x)

## Quick start
Before start, download MIND dataset [here](https://msnews.github.io/#getting-start) and Glove embedding.
```
mkdir data && cd data

# Download MIND dataset
# By downloading the dataset, you agree to the [Microsoft Research License Terms](https://go.microsoft.com/fwlink/?LinkID=206977). For more detail about the dataset, see https://msnews.github.io/.
wget https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip https://mind201910small.blob.core.windows.net/release/MINDlarge_test.zip
unzip MINDlarge_train.zip -d train
unzip MINDlarge_dev.zip -d val
unzip MINDlarge_test.zip -d test
rm MINDlarge_*.zip

# Download GloVe pre-trained word embedding
wget https://nlp.stanford.edu/data/glove.840B.300d.zip
sudo apt install unzip
unzip glove.840B.300d.zip -d glove
rm glove.840B.300d.zip
```

Then, preprocess the data.

```
# Preprocess data into appropriate format
cd ..
python3 data_preprocess.py
```

**Remember you should modify `num_*` in `config.py` by the output of `data_preprocess.py`**

To run the NRMS model using Adam Optimizer, run `python3 Syft_train_individuel.py`.

To run the NRMS model using SGD Optimizer, run `python3 Syft_train.py`.

## More information

Thanks for yusanshi who provides the pytorch version of NewsRecommenders. Discover his work [here](https://github.com/yusanshi/NewsRecommendation).

The official Microsoft News Recommenders Project can be found [here](https://github.com/microsoft/recommenders).