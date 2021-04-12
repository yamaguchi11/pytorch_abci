import os
import time
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.models as models
import sys
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('--epochs',default=1, type=int)
parser.add_argument('--batch-size',default=100, type=int)
parser.add_argument('--optimizer', default="SGD")
parser.add_argument('--momentum', default=0.0, type=float)
parser.add_argument('--beta1', default=0.9, type=float)
parser.add_argument('--beta2', default=0.999, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--weight-decay', default=0.0, type=float)
parser.add_argument('--arch',default="vgg11")
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--amp', action='store_true')
parser.add_argument('--summary', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--p-hori', default=0.0, type=float)
parser.add_argument('--rotation', default=0.0, type=float)
parser.add_argument('--shear', default=0.0, type=float)
parser.add_argument('--min-scale', default=1.0, type=float)
parser.add_argument('--max-scale', default=1.0, type=float)
parser.add_argument('--brightness', default=0.0, type=float)
parser.add_argument('--contrast', default=0.0, type=float)
parser.add_argument('--saturation', default=0.0, type=float)
parser.add_argument('--hue', default=0.0, type=float)
parser.add_argument('--dataset', default="CIFAR10", type=str)

def main():
    args = parser.parse_args()
    epochs=args.epochs
    batch_size = args.batch_size
    print(batch_size)
    rank = int(os.getenv('OMPI_COMM_WORLD_RANK', '0'))
    world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", '1'))
    gpu = rank % torch.cuda.device_count()

    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method = 'tcp://' + master_ip + ':' + master_port
    train(epochs,batch_size,rank,world_size,gpu,master_ip,master_port,init_method,args)

#class LeNet(nn.Module):
class ConvNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
    self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
    self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
    self.fc1 = nn.Linear(4*4*64, 500)
    self.dropout1 = nn.Dropout(0.5)
    self.fc2 = nn.Linear(500, 10)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, 2, 2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2, 2)
    x = F.relu(self.conv3(x))
    x = F.max_pool2d(x, 2, 2)
    x = x.view(-1, 4*4*64)
    x = F.relu(self.fc1(x))
    x = self.dropout1(x)
    x = self.fc2(x)
    return x

def train(epochs,batch_size,rank,world_size,gpu,master_ip,master_port,init_method,args):
    #rank = args.nr * args.gpus + gpu
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', init_method=init_method, world_size=world_size, rank=rank)
    torch.manual_seed(0)
    #model = ConvNet()
    if(args.arch=="ResNet18"):
        model= ResNet18()
    elif(args.arch=="ResNet34"):
        model= ResNet34()
    elif(args.arch=="ResNet50"):
        model= ResNet50()
    elif(args.arch=="ResNet101"):
        model= ResNet101()
    elif(args.arch=="ResNet152"):
        model= ResNet152()
    elif(args.arch=="ResNeXt29_2x64d"):
        model= ResNeXt29_2x64d()
    elif(args.arch=="ResNeXt29_4x64d"):
        model= ResNeXt29_4x64d()
    elif(args.arch=="ResNeXt29_8x64d"):
        model= ResNeXt29_8x64d()
    elif(args.arch=="ResNeXt29_32x4d"):
        model= ResNeXt29_32x4d()
    elif(args.arch=="MobileNetV2"):
        model= MobileNetV2()
    elif(args.arch=="VGG11"):
        model=VGG('VGG11')
    elif(args.arch=="VGG13"):
        model=VGG('VGG13')
    elif(args.arch=="VGG16"):
        model=VGG('VGG16')
    elif(args.arch=="VGG19"):
        model=VGG('VGG19')
    model.to(gpu)

    if(args.summary==True):
        from torchsummary import summary
        if(args.dataset=="CIFAR10"):
            print(args.dataset)
            print(args.arch)
            summary(model,(3,32,32))
        else:
            print(args.dataset)
            print(args.arch)
            summary(model,(3,224,224)) # summary(model,(channels,H,W))
        exit()


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    criterion2 = nn.CrossEntropyLoss(reduction='sum').cuda(gpu)
    # Wrap the model
    # Data loading code
    # Initialization
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    #if rank==0:
    #    wandb.watch(model)
###########################################################################################################################
    if(args.dataset=="CIFAR10"):
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(p=args.p_hori),
                                              transforms.RandomAffine(args.rotation, shear=args.shear, scale=(args.min_scale,args.max_scale)),
                                              transforms.ColorJitter(brightness=args.brightness, contrast=args.contrast, saturation=args.saturation,hue=args.hue),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                             ])

        transform = transforms.Compose([transforms.Resize((32,32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                       ])

        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        validation_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        #training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=100, shuffle=True)
        #validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = 80, shuffle=False)
    ###########################################################################################################################

        #train_dataset = torchvision.datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)
        if(args.test==True):
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=world_size,rank=rank,shuffle=False)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=world_size,rank=rank)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   num_workers=5,
                                                   pin_memory=True,
                                                   sampler=train_sampler)

        validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset,num_replicas=world_size,rank=rank,shuffle=False)
        validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   num_workers=5,
                                                   pin_memory=True,
                                                   sampler=validation_sampler)
    elif(args.dataset=="Imagenet"):
        traindir="/fs2/groups2/gac50562/image-net/train"
        valdir="/fs2/groups2/gac50562/image-net/val"

        transform_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
                                             ])

        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
                                       ])

        train_dataset = torchvision.datasets.ImageFolder(traindir,transform_train)
        validation_dataset = torchvision.datasets.ImageFolder(valdir,transform)
        if(args.test==True):
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=world_size,rank=rank,shuffle=False)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=world_size,rank=rank)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   num_workers=5,
                                                   pin_memory=True,
                                                   sampler=train_sampler)

        validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset,num_replicas=world_size,rank=rank,shuffle=False)
        validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   num_workers=5,
                                                   pin_memory=True,
                                                   sampler=validation_sampler)
################################################################################################################################
    if rank==0:
        import wandb
        run=wandb.init(project="my-project")
        wandb.config.world_size=world_size
        wandb.config.total_batch_size=args.batch_size*world_size
        wandb.config.update(args)


    if(args.test==True):
        test_i=0

    lr=args.lr
    start = datetime.now()
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        if(args.dataset=="CIFAR10"):
            if(epoch==149):
                lr=0.1*lr
            if(epoch==224):
                lr=0.1*lr
        else:
            if(epoch==29):
                lr=0.1*lr
            if(epoch==59):
                lr=0.1*lr
            if(epoch==89):
                lr=0.1*lr
        if(args.optimizer=="SGD"):
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif(args.optimizer=="Adam"):
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,betas=(args.beta1,args.beta2), weight_decay=args.weight_decay)
        elif(args.optimizer=="AdamW"):
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr,betas=(args.beta1,args.beta2), weight_decay=args.weight_decay)
        model.train()
        running_corrects = 0.0
        epoch_acc=0.0
        epoch_loss=0.0
        epoch_loss_validation=0.0
        epoch_acc_validation=0.0
        number_val_data=0
        if(args.amp==True):
            scaler = torch.cuda.amp.GradScaler()

        for i, (images, labels) in enumerate(train_loader):
            if(args.test==True):
                test_i+=1
                #start = datetime.now()
            total_step = len(train_loader)
            images = images.to(gpu,non_blocking=True)
            labels = labels.to(gpu,non_blocking=True)
            # Forward pass
            optimizer.zero_grad()
            if(args.amp==True):
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                # Backward and optimize
                loss.backward()
                optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_corrects_t = torch.sum(preds == labels.data).float()/len(labels.data)
            #if (i + 1) % 10 == 0:
            print('RANK[{}],GPU[{}], Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}'.format(rank,gpu,epoch + 1, epochs, i + 1, total_step,
                                                                         loss.item(),running_corrects_t.item()))


            dist.all_reduce(loss)
            epoch_loss+=loss.item()/world_size
            dist.all_reduce(running_corrects_t)
            epoch_acc+= running_corrects_t.item()/world_size
            if(args.test==True):
                if(test_i>99):
                    break
        if(args.test==True):
            if(test_i>99):
                break
        epoch_acc=epoch_acc/ total_step
        epoch_loss=epoch_loss/total_step

        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(validation_loader):
                number_val_data+=len(labels.data)
                images = images.to(gpu,non_blocking=True)
                labels = labels.to(gpu,non_blocking=True)
                # Forward pass
                if(args.amp==True):
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion2(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion2(outputs, labels)
                _, preds = torch.max(outputs, 1)
                dist.all_reduce(loss)
                epoch_loss_validation+=loss.item()
                epoch_acc_validation_t=torch.sum(preds == labels.data).float()
                dist.all_reduce(epoch_acc_validation_t)
                epoch_acc_validation+= epoch_acc_validation_t.item()
                #print(type(epoch_acc_validation))
        epoch_acc_validation=epoch_acc_validation/(world_size*number_val_data)
        epoch_loss_validation=epoch_loss_validation/(world_size*number_val_data)

        if rank==0:
            wandb.log({'loss_train': epoch_loss,'accuracy_train': epoch_acc, 'loss_val': epoch_loss_validation,'accuracy_val': epoch_acc_validation})
            print('RANK[{}],GPU[{}], Epoch [{}/{}],  Loss: {:.4f}, Acc:{:.4f},  Loss_val: {:.4f}, Acc_val:{:.4f}'.format(rank,gpu, epoch + 1,  epochs, epoch_loss,epoch_acc, epoch_loss_validation,epoch_acc_validation))

    if rank==0:
        training_time=datetime.now() - start
        print("Training complete in: " + str(training_time))
        wandb.config.training_time=training_time
        run.finish()

if __name__ == '__main__':
    main()
    print("Finished")
    #sys.exit(0)
