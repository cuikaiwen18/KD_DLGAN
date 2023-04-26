
import torch
import torch.nn.functional as F

# LeCam Regularziation loss
def lecam_reg(dis_real, dis_fake, ema):
  reg = torch.mean(F.relu(dis_real - ema.D_fake).pow(2)) + \
        torch.mean(F.relu(ema.D_real - dis_fake).pow(2))
  return reg

# ------ non-saturated ------ #
def loss_dcgan_dis(dis_fake, dis_real, ema=None, it=None):
  L1 = torch.mean(F.softplus(-dis_real))
  L2 = torch.mean(F.softplus(dis_fake))
  return L1, L2

def loss_dcgan_gen(dis_fake, dis_real=None):
  loss = torch.mean(F.softplus(-dis_fake))
  return loss

# ------ lsgan ------ #
def loss_ls_dis(dis_fake, dis_real, ema=None, it=None):
  loss_real = torch.mean((dis_real + 1).pow(2))
  loss_fake = torch.mean((dis_fake - 1).pow(2))
  return loss_real, loss_fake

def loss_ls_gen(dis_fake, dis_real=None):
  return torch.mean(dis_fake.pow(2))

# ------ rahinge ------ #
def loss_rahinge_dis(dis_fake, dis_real, ema=None, it=None):
  loss_real = torch.mean(F.relu(1. - (dis_real - torch.mean(dis_fake)))/2)
  loss_fake = torch.mean(F.relu(1. + (dis_fake - torch.mean(dis_real)))/2)
  return loss_real, loss_fake

def loss_rahinge_gen(dis_fake, dis_real):
  if torch.is_tensor(dis_real):
    dis_real = torch.mean(dis_real).item()
  loss = F.relu(1 + (dis_real - torch.mean(dis_fake)))/2 + F.relu(1 - (dis_fake - dis_real))/2
  return torch.mean(loss)

# ------ hinge ------ #
def loss_hinge_dis(dis_fake, dis_real, ema=None, it=None):
  if ema is not None:
    # track the prediction
    ema.update(torch.mean(dis_fake).item(), 'D_fake', it)
    ema.update(torch.mean(dis_real).item(), 'D_real', it)

  loss_real = F.relu(1. - dis_real)
  loss_fake = F.relu(1. + dis_fake)
  return torch.mean(loss_real), torch.mean(loss_fake)

def loss_hinge_gen(dis_fake, dis_real=None):
  loss = -torch.mean(dis_fake)
  return loss


def cls_loss(D_out, y):
    # print(D_real_out_cls.get_device(), y_real.get_device(),D_fake_out_cls.get_device(0),  y_fake.get_device())
    # print('-----------')
    aux_criterion = torch.nn.NLLLoss().cuda()
    cls_errD = aux_criterion(D_out, y)
    return cls_errD

    


    
def TripletLoss_reverse(anchor, pos, neg):

    num_samples = anchor.shape[0]
    y = torch.ones((num_samples, 1)).view(-1).to(anchor.device)
    ap_dist = torch.norm(anchor-pos, 2, dim=1).view(-1)
    an_dist = torch.norm(anchor-neg, 2, dim=1).view(-1)
    # loss = torch.nn.SoftMarginLoss()( -ap_dist, y) + torch.nn.SoftMarginLoss()( -an_dist, y)
    loss = ap_dist.mean() + an_dist.mean()
    return loss

def img_lan_out_distri_loss(f_s, t_s):
    bsz = f_s.shape[0]
    f_s = f_s.view(bsz, -1)
    
    G_s = torch.mm(f_s.half(), torch.t(t_s))
    # G_s = G_s / G_s.norm(2)
    G_s = torch.nn.functional.normalize(G_s)

    idx = torch.randperm(bsz)
    idx2 = torch.range(0,bsz-1)
    G_s_shuffle = G_s[idx]

    attn = (idx != idx2).type(torch.uint8).to(G_s.device)   
    
    # import pdb
    # pdb.set_trace()
    G_diff = (G_s - G_s_shuffle) * attn.unsqueeze(1)
    # print(attn)

    loss = (G_diff * G_diff).view(-1, 1).sum(0)

   
    return loss 

def img_lan_similarity_loss(f_s, t_s, f_t, t_t):
    bsz = f_s.shape[0]
    f_s = f_s.view(bsz, -1)
    f_t = f_t.view(bsz, -1)
    
    G_s = torch.mm(f_s.half(), torch.t(t_s))
    # G_s = G_s / G_s.norm(2)
    G_s = torch.nn.functional.normalize(G_s)
    G_t = torch.mm(f_t.half(), torch.t(t_t))
    # G_t = G_t / G_t.norm(2)
    G_t = torch.nn.functional.normalize(G_t)

    G_diff = G_t - G_s
    loss = (G_diff * G_diff).view(-1, 1).sum(0)
    return loss
# Default to hinge loss
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis

