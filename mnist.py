import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 20
batch_size = 100
learning_rate = 1e-3

# MNIST Dataset
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor(),
                           download=True)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Neural Network Model
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        return out


net = Net(input_size, num_classes)


def train(dataloder = train_loader, model = net, epochNum = num_epochs, learningRate = learning_rate):
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    # Train the Model
    size = len(dataloder.dataset)
    totalLoss = 0
    for epoch in range(epochNum):
        for i, (images, labels) in enumerate(dataloder):
            # Convert torch tensor to Variable
            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)

            # Forward + Backward + Optimize
            pred = model(images)
            loss = criterion(pred, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totalLoss += loss.item()
            # loss, current = loss.item(), i * len(images)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    totalLoss /= size * epochNum
    print(f"The total loss is {totalLoss}")
    return totalLoss

def test(dataloder = test_loader, model = net):
    # Test the Model
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in dataloder:
            images = Variable(images.view(-1, 28*28))
            pred = model(images)
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
            total += labels.size(0)

    acuracy = (100 * correct / total)
    print('Accuracy of the network on the 10000 test images: %d %%' % acuracy)
    return acuracy

# Section a
train(epochNum=100)
test()

# Section b
maxAcuracy = 0
maxLearningRate = None
maxBatchSize = None
for curBatchSize in range(50, 250, 50):
    train_loader_sec2 = torch.utils.data.DataLoader(dataset=train_dataset,
        batch_size=curBatchSize,
        shuffle=True)
    test_loader_sec2 = torch.utils.data.DataLoader(dataset=test_dataset,
        batch_size=curBatchSize,
        shuffle=False)
    for i in range(-5, 5, 1):
        curLearningRate = learning_rate + i * 10e-4
        train(dataloder=train_loader_sec2, learningRate=curLearningRate)
        curAcuracy = test(dataloder=test_loader_sec2)
        if curAcuracy > maxAcuracy:
            maxAcuracy = curAcuracy
            maxBatchSize = curBatchSize
            maxLearningRate = curLearningRate

# The best acuracy is 92.15, the best batch size is 200, the best learning rate is 0.004 
train_loader_sec2 = torch.utils.data.DataLoader(dataset=train_dataset,
    batch_size=maxBatchSize,
    shuffle=True)
test_loader_sec2 = torch.utils.data.DataLoader(dataset=test_dataset,
    batch_size=maxBatchSize,
    shuffle=False)

train(dataloder=train_loader_sec2, epochNum=100, learningRate=maxLearningRate)
test(dataloder=test_loader_sec2)



# Section c
class Sec3Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Sec3Net, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, num_classes),
            nn.ReLU()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model=Sec3Net(input_size, num_classes)
train(model=model, epochNum=100)
test(model=model)




# Save the Model
torch.save(net.state_dict(), 'model.pkl')