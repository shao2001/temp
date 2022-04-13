# pip3 install torch torchvision torchaudio torchinfo
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


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
  "val"

  running_loss = 0.0
  for i, data in tqdm(enumerate(testloader, 0), total=len(testloader), leave=True, position=0):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data

      # zero the parameter gradients

      # forward + backward + optimize
      outputs = vgg16(inputs.to(device))
      loss = criterion(outputs, labels.to(device))

      # print statistics
      running_loss += loss.item()

  print()
  print(f'loss: {running_loss / 2000:.3f}')

  print('Finished Validation')





