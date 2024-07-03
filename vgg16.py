import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

from nbdt.model import SoftNBDT
from nbdt.loss import SoftTreeSupLoss
from nbdt.hierarchy import generate_hierarchy
from nbdt.tree import Tree
from PIL import Image
from utils import train, test, load_vgg16, load_data, plot_decision_tree


def main():
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    model = load_vgg16(num_classes=10).to(device)

    # load data
    classes = (
        'airplane', 'car', 'bird', 'cat', 'deer', 
        'dog', 'frog', 'horse', 'ship', 'truck'
    )
    trainloader, testloader = load_data()

    # pretrain model
    optimizer = torch.optim.SGD(model.classifier[-1].parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    train(model, trainloader, optimizer, nn.CrossEntropyLoss(), epochs=3, device=device)

    # generate hierarchy from pretrained model
    generate_hierarchy(dataset='CIFAR10', arch='vgg16', model=model, method='induced')

    # generate tree visualizer in HTML
    tree = Tree(dataset='CIFAR10', path_graph=None, path_wnids=None, classes=None, hierarchy='induced-vgg16')
    tree.visualize('./out/CIFAR10-induced-VGG16-tree.html', dataset='CIFAR10')

    # fine-tune model with tree supervision loss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    criterion = SoftTreeSupLoss(dataset='CIFAR10', hierarchy='induced-vgg16', criterion=criterion)
    train(model, trainloader, optimizer, criterion, epochs=2, device=device)

    # run inference using embedded decision rules
    model_nbdt = SoftNBDT(model=model, dataset='CIFAR10', hierarchy='induced-vgg16')
    model_nbdt.eval()

    # evaluate on test set
    test_acc = test(model_nbdt, testloader)
    print('Test accuracy:', test_acc)

    # save checkpoint
    torch.save(
        {'state_dict': model_nbdt.state_dict()}, 
        './SoftNBDT_model.pt'
    )


if __name__ == '__main__':
    main()