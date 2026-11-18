from torch import nn
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt 

train_data= datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data= datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

labels_map = {
    0: "T-shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}

figure = plt.figure(figsize=(8,8))
cols,rows = 3, 3
for i in range (1,cols * rows +1):
    sample_idx = torch.randint(len(train_data),size=(1,)).item()
    img,label = train_data[sample_idx]
    figure.add_subplot(rows,cols,i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(),cmap="gray")
plt.show()

batch_size=25

#create data loaders.
train_dataloader= DataLoader(train_data,batch_size=batch_size)
test_dataloader = DataLoader(test_data,batch_size=batch_size)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

#defind model
class NuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten =nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
    def forward(self,x):
        x=self.flatten(x)
        logits=self.linear_relu_stack(x)
        return logits

model= NuralNetwork().to(device)
print(model)

loss_fn=nn.CrossEntropyLoss()
print(list(model.parameters()))
optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)

def train(dataloader,model,loss_fn,optimizer):
    size =len(dataloader.dataset)
    model.train()
    for batch,(X,y) in enumerate(dataloader):
        X,y =X.to(device),y.to(device)

        #compute prediction error
        pred =model(X)
        loss = loss_fn(pred,y)

        #Backpropagarion
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss,current = loss.item(), (batch+1)*len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader,model,loss_fn):
    size= len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct= 0,0
    with torch.no_grad():
        for X,y in dataloader:
            X,y = X.to(device),y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred,y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /=size
    print(f"Test Error: \n Accuracy : {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n ")

epochs =5
for t in range(epochs):
    print(f"Epoch {t+1}\n----------------------------")
    train(train_dataloader,model,loss_fn,optimizer)
    test(test_dataloader,model,loss_fn)
print("Done")

torch.save(model.state_dict(),"model.pth")
print("saved Pytorch Model state to model.pth")

model = NuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))

classes =[
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x,y = test_data[0][0],test_data[0][1]
with torch.no_grad():
    x=x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted : "{predicted}", Actual: "{actual}"')



