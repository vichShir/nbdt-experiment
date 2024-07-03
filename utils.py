import numpy as np
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from nbdt.model import SoftNBDT
from nbdt.loss import SoftTreeSupLoss
from nbdt.hierarchy import generate_hierarchy
from nbdt.tree import Tree
from PIL import Image
from tqdm import tqdm


def train(net, trainloader, optimizer, criterion, epochs=10, device='cuda'):
    net.train()
    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()


def test(net, testloader, device='cuda'):
    net.eval()
    with torch.no_grad():
        pred_classes = []
        test_targets = []
        for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = torch.argmax(net(inputs), dim=1)
            pred_classes.append(outputs)
            test_targets.append(targets)

        acc = accuracy_score(torch.cat(test_targets).cpu().numpy(), torch.cat(pred_classes).cpu().numpy())
        return acc


def load_vgg16(num_classes=1000):
    # load pretrained model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    # modify last linear layer
    mod = list(model.classifier.children())
    mod.pop()
    mod.append(torch.nn.Linear(4096, num_classes))
    new_classifier = torch.nn.Sequential(*mod)
    model.classifier = new_classifier
    return model


def load_data():
    transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=False, 
        transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=128,
        shuffle=True, 
        num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=128,
        shuffle=False, 
        num_workers=2
    )
    return (trainloader, testloader)


def plot_decision_tree(decisions, nbdt):
    graph = nbdt.rules.tree.G
    
    labeldict = {}
    for wnid, node in nbdt.rules.tree.wnid_to_node.items():
        labeldict[wnid] = node.name

    color_map = []
    for node_name in list(labeldict.values()):
        if node_name in list(map(lambda x: x['name'], decisions)):
            color_map.append('royalblue')
        else:
            color_map.append('whitesmoke')

    pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
    nx.draw(graph, pos, 
            labels=labeldict, 
            with_labels=True, 
            node_color=color_map, 
            node_size=1400, 
            font_color='black',
            font_weight='bold',
            font_family='sans-serif',
            font_size=8,
            edge_color='lightgray')
    
    path_pos = [pos[n['node'].wnid] for n in decisions[:-1]]
    for idx, d in enumerate(decisions[1:]):
        x, y = path_pos[idx]
        prob = d['prob']
        plt.text(x+5, y-30, s=f'Prob. {prob:.0%}', 
                 horizontalalignment='center', 
                 fontsize='x-small', 
                 color='darkcyan', 
                 fontweight='bold')
    
    plt.show()