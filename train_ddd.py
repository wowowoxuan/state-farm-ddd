import torch
import torchvision
from torchvision import transforms as T
import os
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from dataset.Dataset_v1 import train_val_set
from models.mobilenetv2 import MobileNetV2,tinyNet, VGG16, AlexNet, ResNet18

val_fold_idx = 0
Exp_name='use_pretrained'
net = MobileNetV2(n_classes = 10, use_pretrained = True)
# net = tinyNet()
# for item in net.base_model.parameters():
#     item.requires_grad = False
net.cuda().train()    
Exp_path=os.path.join('./TB_file',Exp_name)
if not os.path.exists(Exp_path):
    os.mkdir(Exp_path)
TB_path=os.path.join(Exp_path,'TB')
if not os.path.exists(TB_path):
    os.mkdir(TB_path)
tensorboard=SummaryWriter(TB_path)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode = 'min', factor = 0.1, patience = 10, verbose = True, cooldown = 0)
train_transform = T.Compose([
    T.Resize((224,224)),
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.RandomAffine(15),
    T.ToTensor()
])

test_transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),

])

trainset = torchvision.datasets.ImageFolder(root = '/media/weiheng/Elements/Data/state-farm-distracted-driver-detection/imgs/train',transform=train_transform)
testset = torchvision.datasets.ImageFolder(root = '/media/weiheng/Elements/Data/state-farm-distracted-driver-detection/imgs/test',transform=test_transform)
print(len(testset))
train_loader = torch.utils.data.DataLoader(dataset = trainset, batch_size = 128, shuffle = True, num_workers = 2)
val_loader = torch.utils.data.DataLoader(dataset = testset, batch_size = 128, shuffle = False, num_workers = 2)
best_acc = 0
for epoch in range(200):
    print('epoch:' + str(epoch))
    epo_summary={}
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx,(train_x,train_y) in tqdm(enumerate(train_loader)):
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        optimizer.zero_grad()
        outputs = net(train_x)
        loss = criterion(outputs,train_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _,predicted = outputs.max(1)
        total += train_y.size(0)
        correct += predicted.eq(train_y).sum().item()
    print('train_acc:' + str(100*correct/total) + '%')
    ep_train_loss=train_loss/(batch_idx+1)
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx,(val_x,val_y) in tqdm(enumerate(val_loader)):
            val_x = val_x.cuda()
            val_y = val_y.cuda()
            outputs = net(val_x)
            loss = criterion(outputs,val_y)
            test_loss += loss.item()
            _,predicted = outputs.max(1)
            total += val_y.size(0)
            correct += predicted.eq(val_y).sum().item()

    acc = 100.*correct/total
    print('val_acc:' + str(acc) + '%')
    ep_valid_loss=test_loss/(batch_idx+1)
    scheduler.step(ep_valid_loss)
    epo_summary['loss/train']=ep_train_loss
    epo_summary['loss/valid']=ep_valid_loss
    epo_summary['accuracy']=acc
    for name,val in epo_summary.items():
        tensorboard.add_scalar(name,val,epoch)
    if best_acc < acc or best_acc == acc:
        print('save model..................................................')
        best_acc = acc
        torch.save(net.state_dict(), './mobilenetv2_random_split.pth')
    print(best_acc)