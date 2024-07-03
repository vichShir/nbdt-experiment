import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from nbdt.model import SoftNBDT
from nbdt.loss import SoftTreeSupLoss
from nbdt.hierarchy import generate_hierarchy
from nbdt.tree import Tree
from tqdm import tqdm


def pretrain(net, trainloader, optimizer, criterion, epochs=10, device='cuda'):
    net.train()
    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()


def train_nbdt(net, trainloader, optimizer, criterion, epochs=10, device='cuda'):
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
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    mod = list(model.classifier.children())
    mod.pop()
    mod.append(torch.nn.Linear(4096, num_classes))
    new_classifier = torch.nn.Sequential(*mod)
    model.classifier = new_classifier
    return model


def load_data():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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


def main():
    torch.manual_seed(0)

    # load model
    model = load_vgg16(num_classes=10)
    model = model.cuda()

    # load data
    classes = (
        'airplane', 'car', 'bird', 'cat', 'deer', 
        'dog', 'frog', 'horse', 'ship', 'truck'
    )
    trainloader, testloader = load_data()

    # pretrain model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    pretrain(model, trainloader, optimizer, nn.CrossEntropyLoss(), epochs=5)

    # generate hierarchy from pretrained model
    generate_hierarchy(dataset='CIFAR10', arch='vgg16', model=model, method='induced')

    # generate tree visualizer in HTML
    tree = Tree(dataset='CIFAR10', path_graph=None, path_wnids=None, classes=None, hierarchy='induced-vgg16')
    tree.visualize('./out/CIFAR10-induced-VGG16-tree.html', dataset='CIFAR10')

    # fine-tune model with tree supervision loss
    criterion = nn.CrossEntropyLoss()
    criterion = SoftTreeSupLoss(dataset='CIFAR10', hierarchy='induced-vgg16', criterion=criterion)
    train_nbdt(model, trainloader, optimizer, criterion, epochs=20)

    # run inference using embedded decision rules
    model = SoftNBDT(model=model, dataset='CIFAR10', hierarchy='induced-vgg16')
    model.eval()

    # test set
    test_acc = test(model, testloader)
    print('Test accuracy:', test_acc)

    # individual sample with decisions
    b = next(iter(testloader))
    X = b[0][0].unsqueeze(0).to('cuda')
    y = b[1][0]

    # print(model.forward_with_decisions(X))
    print('true label:', classes[y])

    for node in model.forward_with_decisions(X)[1][0]:
        print(node)


if __name__ == '__main__':
    main()