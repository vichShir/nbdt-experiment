# Neural-Backed Decision Trees VGG16 & CIFAR10
A experiment with Neural-Backed Decision Trees using VGG16 as backbone and the CIFAR10 dataset. Original [code](https://github.com/alvinwan/neural-backed-decision-trees/tree/master) and [paper](https://arxiv.org/abs/2004.00221).


## Python Environment

### Install packages
```shell
conda create -n nbdt python=3.9.19
conda activate nbdt

conda install pytorch=2.0.0 cudatoolkit=11.8 -c pytorch
conda install -c conda-forge pygraphviz

pip install torchvision==0.15.1 numpy==1.26.4 pytorchcv pyparsing zipp nltk scikit-learn networkx pytest opencv-python==4.4.0.46 matplotlib tqdm gradio notebook==6.0.0
```

### Install NBDT from source
```shell
git clone https://github.com/alvinwan/neural-backed-decision-trees.git
cd neural-backed-decision-trees/
cp ../requirements.txt requirements.txt
pip install .
```

### Setup NBDT installation
```shell
cd nbdt
NBDT_PATH=$(python -c "import nbdt; print(nbdt.__path__[0])")
cp -r thirdparty $NBDT_PATH  # copy thirdparty directory to installation path

cd ../../
rm -rf neural-backed-decision-trees/

cp graph-induced-vgg16.json $NBDT_PATH/hierarchies/CIFAR10  # copy induced hierarchy for VGG16
```

## Model

I pretrained a `VGG16` and `CIFAR10` dataset for `1 epoch` to build the induced hierarchy at `./graph-induced-vgg16.json`.

A `Soft NBDT` was also trained with the built induced hierarchy for `2 epochs` as in `./vgg16.py`. The checkpoint can be found [here](https://drive.google.com/file/d/1vXoSaZ2lM9cEojcJN-EcapUE0tabCsjs/view?usp=sharing).

## Generate a new Induced Hierarchy

Edit in `$NBDT_PATH/graph.py` to recognize the last layer of the VGG16 model, which is the `classifier.6` layer.
```python
################
# INDUCED TREE #
################


MODEL_FC_KEYS = (
    "fc.weight",
    "linear.weight",
    "module.linear.weight",
    "module.net.linear.weight",
    "output.weight",
    "module.output.weight",
    "output.fc.weight",
    "module.output.fc.weight",
    "classifier.weight",
    "classifier.6.weight",  # add this line
    "model.last_layer.3.weight",
)
```

Make sure to be in the `nbdt-experiment` folder that you cloned. Then, you have to copy the `wnids` folder located in pip directories to your local `./nbdt/wnids`.
```shell
cd nbdt-experiment  # make sure to be in the correct folder
mkdir ./nbdt/wnids
cp $NBDT_PATH/wnids/CIFAR10.txt ./nbdt/wnids/
```

After you generate the Induced Hierarchy, it will save at `./nbdt/hierarchies/CIFAR10`. Make sure to copy the generated induced hierarchy to the package installation directory.
```shell
cp ./nbdt/hierarchies/CIFAR10/graph-induced-vgg16.json $NBDT_PATH/hierarchies/CIFAR10/
```
