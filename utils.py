import torch
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
def L_infinity(t1, t2):
    return torch.max(torch.abs(t1-t2)).item()

def fix(img_1, img_2, v):
    differences = img_2 - img_1
    distances = torch.abs(differences)
    mask = distances > v
    sign = torch.sign(differences)
    tmp = img_1 + sign * v
    new_t = torch.where(mask, tmp, img_2)
    return new_t

def denormalize(img, mean, std):
    mean = torch.tensor(mean, device=device).view(1,3,1,1)
    std = torch.tensor(std, device=device).view(1,3,1,1)

    return img * std + mean
