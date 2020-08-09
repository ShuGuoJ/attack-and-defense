import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from utils import L_infinity, fix, denormalize
from visdom import Visdom
from matplotlib import pyplot as plt

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model_path = 'model/resnet18_2.pkl'
net = models.resnet18()
net.fc = nn.Linear(512, 2)
net.load_state_dict(torch.load(model_path))
net.to(device)
criterion = nn.CrossEntropyLoss()
criterion.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
])

img_path = 'data/attack.jpg'
img = Image.open(img_path)
input = transform(img)
input = input.unsqueeze(0).to(device)
input.requires_grad_()
label = torch.LongTensor([1]).to(device)
net.eval()
epoch = 150
lr = 10
threshold = 0.01
viz = Visdom()
viz.image(denormalize(input, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])[0], win='raw')
viz.line([0.], [0.], win='loss', opts=dict(title='loss',
                                           legend=['loss']))
for i in range(epoch):
    input = input.to(device)
    logits = net(input)
    loss = - criterion(logits, label)

    loss.backward()
    new_input = input - lr * input.grad

    if L_infinity(input, new_input) > threshold:
        new_input = fix(input, new_input, threshold)
    input = new_input.detach()
    input.requires_grad_()
    viz.line([loss.item()], [epoch], win='loss', update='append')

input = input - torch.min(input)
input = input / torch.max(input)
logits = net(input)
pred = torch.softmax(logits, dim=-1).detach().cpu().numpy()
label = torch.argmax(logits, dim=-1)[0].item()
print('dog') if label==1 else print('cat')
viz.image(input[0],win='attack')
plt.bar(list(range(pred.shape[1])), pred[0], tick_label=['cat', 'dog'])
plt.savefig('probability_distribution.jpg')