import torch
from visdom import Visdom
from torchvision import models
from torch import optim
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
net = models.vgg16(pretrained=True)
net.to(device)
input = torch.rand((1,3,224,224), requires_grad=True, device=device)
lr = 1e-1
epochs = 50
net.eval()
label = 0
viz = Visdom()
viz.line([[0.,0.]], [0.], win='loss&&probability', opts=dict(title='loss&&probability',
                                           legend=['loss', 'probability']))
for i in range(epochs):
    out = net(input)
    loss = -out[0, label]
    loss.backward()
    input = input - lr * input.grad
    input = input - torch.max(input)
    input = input / torch.min(input)
    input = input.detach()
    input.requires_grad_()
    pred = torch.softmax(out, dim=-1)
    print('epoch:{} loss:{} probability:{}'.format(i, loss, pred[0, label]))
    viz.line([[loss.item(), pred[0,label].item()]], [i], win='loss&&probability', update='append')

viz.images(input, win='input')
print(torch.max(input), torch.min(input))