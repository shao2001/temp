# pip3 install torch torchvision torchaudio torchinfo
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchinfo
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def train(model, loss_fn, optimizer, dataloader, epoch_num=100, device='cuda:0'):
  model.train()
  for epoch in range(epoch_num):  # loop over the dataset multiple times
    print(f'epoch = {epoch}')
    for data in tqdm(dataloader, total=len(dataloader), leave=True, position=0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

  print('Finished Training')


def val(model, loss_fn, dataloader, device='cuda:0'):
  model.eval()
  acc_loss = 0.0
  for _, data in tqdm(enumerate(dataloader), total=len(dataloader), leave=True, position=0):
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    acc_loss += loss

  print(acc_loss)
  return acc_loss


if __name__ == '__main__':
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)

  transform = transforms.Compose(
      [transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  # load data
  batch_size = 45
  trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

  classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  # load model
  vgg16 = models.vgg16(pretrained=True)
  vgg16 = vgg16.to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)

  """报错
  RuntimeError: CUDA out of memory. Tried to allocate 552.00 MiB (GPU 0; 14.76 GiB total capacity;
  12.86 GiB already allocated; 277.75 MiB free; 13.17 GiB reserved in total by PyTorch) If reserved
  memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation
  for Memory Management and PYTORCH_CUDA_ALLOC_CONF.  
  """
  val(vgg16, criterion, testloader)




