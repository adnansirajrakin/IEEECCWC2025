import argparse
import os
import time
import logging
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import models_bnn
import models
import numpy as np
from torch.autograd import Variable
from utils.options import args
from utils.common import *
from modules import *
from datetime import datetime 
import dataset
import copy
from thop import profile

import operator


class BFA(object):
    def __init__(self, criterion, k_top=10):

        self.criterion = criterion
        # init a loss_dict to log the loss w.r.t each layer
        self.loss_dict = {}
        self.bit_counter = 0
        self.k_top = k_top
        self.n_bits2flip = 0
        self.loss = 0

    def flip_bit(self, m):
        
        # 1. flatten the gradient tensor to perform topk
        self.k_top = m.weight.view(-1).size()[0]
        w_grad_topk, w_idx_topk = m.weight.grad.detach().abs().view(-1).topk(
            self.k_top)
        # update the b_grad to its signed representation
        w_grad_topk = m.weight.grad.detach().view(-1)[w_idx_topk]
        copy_weight = m.weight.data.detach().clone()

        # self.n_bits2flip loop 
        tracker=0
        i=0
        while tracker < self.n_bits2flip :
            # top1 check gradient  ++  dont
            # +- flip
            if(w_grad_topk[i].sign()+m.weight.data.view(-1)[w_idx_topk[i]].sign()!=0):
                # logging.info(w_grad_topk[i].sign().item(),m.weight.data.view(-1)[w_idx_topk[i]].sign().item())
                # logging.info('before: ')
                # logging.info(m.weight.data.view(-1)[w_idx_topk[i]].item())        
                copy_weight.view(-1)[w_idx_topk[i]] = copy_weight.view(-1)[w_idx_topk[i]]*(-1.0)
                tracker+=1
                # logging.info('after: ')
                # logging.info(m.weight.data.view(-1)[w_idx_topk[i]].item())
            i+=1
            # logging.info(tracker)
        ww= copy_weight - m.weight.data
        
        return copy_weight

    def progressive_bit_search(self, model, data, target):
        
        model.eval()

        # 1. perform the inference w.r.t given data and target
        output = model(data)
        target[:] = 3
        self.loss = self.criterion(output, target)
        # 2. zero out the grads first, then get the grads
        for m in model.modules():
            if isinstance(m, BinarizeConv2d) or isinstance(m, bilinear):
                if m.weight.grad is not None:
                    m.weight.grad.data.zero_()

        self.loss.backward()
        # init the loss_max to enable the while loop
        self.loss_max = self.loss.item()

        # 3. for each layer flip #bits = self.bits2flip
        while self.loss_max >= self.loss.item() and self.n_bits2flip<20 :
            
            self.n_bits2flip += 1
            # iterate all the quantized conv and linear layer
            n=0
                  
            for name, module in model.named_modules():
                if isinstance(module, BinarizeConv2d) or isinstance(module, bilinear):
                    n=n+1
                    #print(n,name)
                    clean_weight = module.weight.data.detach()
                    attack_weight = self.flip_bit(module)
                
                    # change the weight to attacked weight and get loss
                    module.weight.data = attack_weight
                    output = model(data)
                    # logging.info(name)
                    new_loss = self.criterion(output,target).item()
                    # if(new_loss!=self.loss_dict[name]):
                        # logging.info(new_loss)
                    self.loss_dict[name] = new_loss
                    # logging.info(self.loss_dict[name])                                      
                    # change the weight back to the clean weight
                    module.weight.data = clean_weight
            
            # after going through all the layer, now we find the layer with max loss
            max_loss_module = min(self.loss_dict.items(),
                                  key=operator.itemgetter(1))[0]
            self.loss_max = self.loss_dict[max_loss_module]
        print(self.loss_dict.items())    
            # logging.info("loss dict: ", self.loss_dict) 
        # 4. if the loss_max does lead to the degradation compared to the self.loss,
        # then change the that layer's weight without putting back the clean weight
        
        
        n=0
        for name, module in model.named_modules():
            n=n+1
            #print(n,name)
            if name == max_loss_module:
                attack_weight = self.flip_bit(module)
                module.weight.data = attack_weight

        # reset the bits2flip back to 0
        self.bit_counter += self.n_bits2flip
        print(self.n_bits2flip)
        self.n_bits2flip = 0

        return

def main():
    global args, best_prec1, conv_modules
    best_prec1 = 0

    random.seed(args.seed)
    if args.evaluate:
        args.results_dir = '/tmp'
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not args.resume:
        with open(os.path.join(save_path,'config.txt'), 'w') as args_file:
            args_file.write(str(datetime.now())+'\n\n')
            for args_n,args_v in args.__dict__.items():
                args_v = '' if not args_v and not isinstance(args_v,int) else args_v
                args_file.write(str(args_n)+':  '+str(args_v)+'\n')

        setup_logging(os.path.join(save_path, 'logger.log'))
        logging.info("saving to %s", save_path)
        logging.debug("run arguments: %s", args)
    else: 
        setup_logging(os.path.join(save_path, 'logger.log'), filemode='a')
    round=1
    if round ==1:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if 'cuda' in args.type:
            args.gpus = [int(i) for i in args.gpus.split(',')]
            cudnn.benchmark = True
        else:
            args.gpus = None
    

    if args.dataset=='tinyimagenet':
        num_classes=200
        model_zoo = 'models.'
    elif args.dataset=='imagenet':
        num_classes=1000
        model_zoo = 'models.'
    elif args.dataset=='cifar10': 
        num_classes=10
        model_zoo = 'models_bnn.'
    elif args.dataset=='cifar100': 
        num_classes=100
        model_zoo = 'models_bnn.'
    if round ==1:
        #model = nn.DataParallel(eval(model_zoo+args.model)(num_classes=num_classes,channels=[384,384,768,768,1536,1536]))
        #[3rd,5th,7,9th,11th,13,14,16,18,20,21]
        
        channels = [16,16,16,16,16,16,16,32,32,32,32,32,32,64,64,64,64,64,64]
        muls = [1,1,1,1,1,1,1,0.96,0.98,1,1,0.98,0.98,0.96,0.98,0.92,0.93,0.88,0.94,0.84]
       
        for i in range(len(channels)):
            channels[i] = int(channels[i]*muls[i]*3)
        model = (eval(model_zoo+args.model)(num_classes=num_classes)).cuda()
       
 
        
    if not args.resume:
        logging.info("creating model %s", args.model)
        logging.info("model structure: %s", model)
        num_parameters = sum([l.nelement() for l in model.parameters()])
        logging.info("number of parameters: %d", num_parameters)

    # evaluate
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            logging.error('invalid checkpoint: {}'.format(args.evaluate))
        else: 
            checkpoint = torch.load(args.evaluate)
            if len(args.gpus)>1:
                checkpoint['state_dict'] = dataset.add_module_fromdict(checkpoint['state_dict'])
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                        args.evaluate, checkpoint['epoch'])
    elif args.resume:
        checkpoint_file = os.path.join(save_path,'checkpoint.pth.tar')
        if os.path.isdir(checkpoint_file):
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            if len(args.gpus)>1:
                checkpoint['state_dict'] = dataset.add_module_fromdict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch'] - 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    criterion = nn.CrossEntropyLoss().cuda()
    criterion = criterion.type(args.type)
    attacker = BFA(criterion)
    if round ==1:
        model = model.type(args.type)
   
    for name, param in model.named_modules():
        if isinstance(param,BinarizeConv2d):
            print(param.weight.size())
    import copy
    
        
    if args.evaluate:
        model.load_state_dict(torch.load('./models/res20_rabnn.pkl'))
        model1 = copy.deepcopy(model)
        val_loader = dataset.load_data(
                    type='val',
                    dataset=args.dataset, 
                    data_path=args.data_path,
                    batch_size=args.batch_size, 
                    batch_size_test=args.batch_size_test, 
                    num_workers=args.workers)
        for batch_idx, (data, target) in enumerate(val_loader):
        	x,y = data.cuda(), target.cuda()
        	#_,p=model(x).data.max(1) 
        	#y=p
        #plt.hist(y.cpu().numpy().flatten())
        #plt.show()
        	break
        epochs = 5000
        for i in range(epochs):
            print(attacker.bit_counter)
            attacker.progressive_bit_search(model, x, y)
            with torch.no_grad():
                val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, 0)
            logging.info('\n Validation Loss {val_loss:.4f} \t'
                        'Validation Prec@1 {val_prec1:.3f} \t'
                        'Validation Prec@5 {val_prec5:.3f} \n'
                        .format(val_loss=val_loss, val_prec1=val_prec1, val_prec5=val_prec5))
            if val_prec1 < 11.0:
                break
        count = 0
        flip_count = 0
        for name, param in model.named_modules():
            if isinstance(param,BinarizeConv2d) or isinstance(param,bilinear):
                count += 1
                count1 =0
                for name1, param1 in model1.named_modules():
                    if isinstance(param1,BinarizeConv2d) or isinstance(param1,bilinear):
                        count1 += 1
                        if count == count1:
                            dff =(param.weight.view(-1)-param1.weight.view(-1))
                            flip_count += dff[dff != 0].size()[0]
                            print(dff[dff != 0].size())
        print(flip_count," Bits were flipped")
        return

    if args.dataset=='imagenet':
        train_loader = dataset.get_imagenet(
                        type='train',
                        image_dir=args.data_path,
                        batch_size=args.batch_size,
                        num_threads=args.workers,
                        crop=224,
                        device_id='cuda:0',
                        num_gpus=1)
        val_loader = dataset.get_imagenet(
                        type='val',
                        image_dir=args.data_path,
                        batch_size=args.batch_size_test,
                        num_threads=args.workers,
                        crop=224,
                        device_id='cuda:0',
                        num_gpus=1)
    else: 
        train_loader, val_loader = dataset.load_data(
                                    dataset=args.dataset, 
                                    data_path=args.data_path,
                                    batch_size=args.batch_size, 
                                    batch_size_test=args.batch_size_test, 
                                    num_workers=args.workers)

    optimizer = torch.optim.SGD([{'params':model.parameters(),'initial_lr':args.lr}], args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    def cosin(i,T,emin=0,emax=0.01):
        "customized cos-lr"
        return emin+(emax-emin)/2 * (1+np.cos(i*np.pi/T))

    if args.resume:
        for param_group in optimizer.param_groups:
            param_group['lr'] = cosin(args.start_epoch-args.warm_up*4, args.epochs-args.warm_up*4,0, args.lr)
    if args.lr_type == 'cos':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-args.warm_up*4, eta_min = 0, last_epoch=args.start_epoch-args.warm_up*4)
    elif args.lr_type == 'step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_decay_step, gamma=0.1, last_epoch=-1)
    if not args.resume:
        logging.info("criterion: %s", criterion)
        logging.info('scheduler: %s', lr_scheduler)

    def cpt_tk(epoch):
        "compute t&k in back-propagation"
        T_min, T_max = torch.tensor(args.Tmin).float(), torch.tensor(args.Tmax).float()
        Tmin, Tmax = torch.log10(T_min), torch.log10(T_max)
        t = torch.tensor([torch.pow(torch.tensor(10.), Tmin + (Tmax - Tmin) / args.epochs * epoch)]).float()
        k = max(1/t,torch.tensor(1.)).float()
        return t, k

    #* setup conv_modules.epoch
    conv_modules=[]
    for name,module in model.named_modules():
        if isinstance(module,nn.Conv2d):
            conv_modules.append(module)
    
    
    '''for name, param in model.named_parameters():
        if 'conv' in name and 'weight' in name and 'bn' not in name:

            if param.data.size()[0] != 10:
                #print(name)
                print(param.data.size())
    print("hi")
    count= 0
    for name, param in model.named_parameters():
        if 'conv' in name and 'weight' in name and 'bn' not in name:

            if param.data.size()[0] != 10:
                #print(name)
                #print(param.data.size())
                if count == 0:
                    count =1
                    param = torch.cat((param.data, torch.zeros(1, 1, 3, 3)), dim=0)
                else:
                    param = nn.Parameter(torch.zeros([int(param.data.size()[0]*2),int(param.data.size()[1]*2),int(param.data.size()[2]),int(param.data.size()[3])]),requires_grad=True)
           
    for name, param in model.named_parameters():
        if 'conv' in name and 'weight' in name and 'bn' not in name:

            if param.data.size()[0] != 10:
                #print(name)
                print(param.data.size()) '''           
               

    with torch.no_grad():
        for module in conv_modules:
            module.epoch = -1
        val_loss, val_prec1, val_prec5 = validate(
            val_loader, model, criterion, 0)
    
    

    if round == 1:

        epochss = 1000
    else:
        epochss = args.epochs
    args.lr = 0.1
    
    
    for epoch in range(args.start_epoch+1, epochss):
        time_start = datetime.now()
        #*warm up
        if args.warm_up and epoch <5:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * (epoch+1) / 5
        for param_group in optimizer.param_groups:
            logging.info('lr: %s', param_group['lr'])

        #* compute t/k in back-propagation
        t,k = cpt_tk(epoch)
        for name,module in model.named_modules():
            if isinstance(module,nn.Conv2d):
                module.k = k.cuda()
                module.t = t.cuda()
        for module in conv_modules:
            module.epoch = epoch
        # train
        train_loss, train_prec1, train_prec5 = train(
            train_loader, model, criterion, epoch, optimizer,round)

        #* adjust Lr
        if epoch >= 4 * args.warm_up:
            lr_scheduler.step()

        # evaluate 
        with torch.no_grad():
            for module in conv_modules:
                module.epoch = -1
            val_loss, val_prec1, val_prec5 = validate(
                val_loader, model, criterion, epoch,round = 2)
            
        # remember best prec
        is_best = val_prec1 > best_prec1
        if is_best:
            #torch.save(model.state_dict(), './models/res20_v2fl.pkl') 
            best_prec1 = max(val_prec1, best_prec1)
            best_epoch = epoch
            best_loss = val_loss

        # save model
        if epoch % 1 == 0:
            '''model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()
            model_parameters = model.module.parameters() if len(args.gpus) > 1 else model.parameters()
            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.model,
                'state_dict': model_state_dict,
                'best_prec1': best_prec1,
                'parameters': list(model_parameters),
            }, is_best, path=save_path)'''

        if args.time_estimate > 0 and epoch % args.time_estimate==0:
           time_end = datetime.now()
           cost_time,finish_time = get_time(time_end-time_start,epoch,args.epochs)
           logging.info('Time cost: '+cost_time+'\t'
                        'Time of Finish: '+finish_time)

        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Training Prec@5 {train_prec5:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_prec5=train_prec5, val_prec5=val_prec5))


    logging.info('*'*50+'DONE'+'*'*50)
    logging.info('\n Best_Epoch: {0}\t'
                     'Best_Prec1 {prec1:.4f} \t'
                     'Best_Loss {loss:.3f} \t'
                     .format(best_epoch+1, prec1=best_prec1, loss=best_loss))
    
    return 

def forward(data_loader, model, criterion, epoch=0, round =1,training=True, optimizer=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for name, param in model.named_parameters():
        if 'conv' in name and 'weight' in name:
            if round == 1:
                print((param.data).size())
            
    end = time.time()
    round = 2
    if round ==2 :
        for name, param in model.named_parameters():
            if 'masking_enable' in name:
                with torch.no_grad():
                    param.data[:] =0 
    for i, (inputs, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if i==1 and training:
            for module in conv_modules:
                module.epoch=-1
        if args.gpus is not None:
            inputs = inputs.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        input_var = Variable(inputs.type(args.type))
        target_var = Variable(target)
        
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        
                    
        
        if type(output) is list:
            output = output[0]

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if training:
            # compute gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, i, len(data_loader),
                             phase='TRAINING' if training else 'EVALUATING',
                             batch_time=batch_time,
                             data_time=data_time, loss=losses,
                             top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg

class BinaryQuantize_m2(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = (torch.sign(input)+1)/2
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        
        grad_input =  grad_output.clone()/2
        return grad_input/2
def train(data_loader, model, criterion, epoch, optimizer,round):
    model.train()
    return forward(data_loader, model, criterion, epoch,round=round,
                   training=True, optimizer=optimizer)


def validate(data_loader, model, criterion, epoch,round =1):
    model.eval()
    return forward(data_loader, model, criterion, epoch,round=round,
                   training=False, optimizer=None)


if __name__ == '__main__':
    main()
 
