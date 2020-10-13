import os
import time
import csv
import random
import glob

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import pandas as pd
import numpy as np
import sklearn
import sklearn.metrics as sklm
from tqdm import tqdm
from PIL import Image

# from models.densenet_models import DenseNet, densenet_loss
from efficientnet_pytorch import EfficientNet
from dataloader.dataloader import nih_dataloader

from utils.utils import AverageMeter, make_dir
from utils.gradcam import GradCam
from utils.misc_functions import apply_colormap

cudnn.benchmark = True

class efficientnet_agent():
    def __init__(self, config):
        self.config = config

        #dataloader
        self.train_dataloader = nih_dataloader(config, train="train")
        self.val_dataloader = nih_dataloader(config, train="val")
        self.test_dataloader = nih_dataloader(config, train="test")
        
        #model (densenet 201)
        self.num_classes_num = len(config.num_classes)
        # self.net = DenseNet(
        #         growth_rate = config.growth_rate,
        #         block_config = (6, 12, 48, 32),
        #         num_init_features = config.num_init_features,
        #         bn_size = config.batch_size,
        #         drop_rate = config.drop_rate,
        #         num_classes = self.num_classes_num
        #         )
        self.detail = 'efficientnet-' + config.effnet_detail
        if config.pretrain == True :
            self.net=EfficientNet.from_pretrained(self.detail, batch_norm_momentum=0.99, num_classes=self.num_classes_num)
        else:
            self.net=EfficientNet.from_name(self.detail, batch_norm_momentum=0.99, num_classes=self.num_classes_num)
        
        self.num_init_ft = self.net._fc.in_features
        # self.net._fc = nn.Linear(self.num_init_ft, self.num_classes_num)
        self.net._fc = nn.Sequential(nn.Linear(self.num_init_ft, self.num_classes_num), nn.Sigmoid())

        self.Pred_label = config.num_classes
        #loss 
        # self.loss = densenet_loss()
        self.loss = nn.BCELoss()
        
        self.val_best_loss = 999999999

        #optimizer
        # self.optim_net = torch.optim.RMSprop(self.net.parameters(), lr=0.256, weight_decay=1e-5, momentum=0.9)
        self.optim_net = torch.optim.Adam(self.net.parameters(), \
                lr=config.lr, betas=(self.config.beta1, self.config.beta2))

        # self.scheduler = \
        #     optim.lr_scheduler.CosineAnnealingLR(self.optim_net, T_max = config.max_epoch)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optim_net, patience=3)        
        #initialize counter
        self.current_epoch = 1
        self.current_iteration = 0

        #cuda, seed flag
        self.is_cuda = torch.cuda.is_available()
        self.cuda = self.is_cuda & config.cuda
        self.manual_seed = random.randint(1, 10000)
        self.manual_seed = config.seed
        random.seed(self.manual_seed)
        self.num_classes = config.num_classes

        if self.cuda:
            self.device = torch.device("cuda")
            torch.cuda.set_device(config.gpu_device)
            torch.cuda.manual_seed_all(config.seed)
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(config.seed)
 
        self.net = self.net.to(self.device)
        self.loss = self.loss.to(self.device)

        #Colums to use as the row labels of the DataFrame, given index_col
        self.df = pd.read_csv(config.label_data_path, index_col=0)

        #make dirs
        self.experiments = os.getcwd() + '/experiments/' + config.exp_model + '/' + config.exp_model_detail + '/'

        self.checkpoint_path = self.experiments + 'checkpoint/'
        self.checkpoint_current = self.checkpoint_path + 'checkpoint.pth.tar'
        self.checkpoint_best = self.checkpoint_path + 'checkpoint_best.pth.tar'

        self.log_path = self.experiments + 'log_path/'
        self.log_path_train = self.log_path + 'train_loss_log.csv'
        self.log_path_val = self.log_path + 'val_loss_log.csv'

        self.result_path = self.experiments + 'result/'
        self.result_pred_path = self.result_path + 'preds.csv'
        self.result_auc_path = self.result_path + 'auc.csv'

        #visual
        self.input_path = config.input_path        
        self.output_path = config.output_path + 'image'
        self.output_pred_path = config.output_path + '/pred/'
        self.model_path = config.model_path
        self.generate_num = config.generate_num

        make_dir(self.checkpoint_path)
        make_dir(self.log_path)
        make_dir(self.result_path)

        make_dir(self.output_path)
        make_dir(self.output_pred_path)

    def save_checkpoint(self, current = True):
        state = {
                'epoch': self.current_epoch,
                'iteration': self.current_iteration,
                'net_state_dict': self.net.state_dict(),
                'net_optimizer': self.optim_net.state_dict(),
                'seed': self.manual_seed,
            }
        # Save the state
        if current:
            torch.save(state, self.checkpoint_current)
        else:
            torch.save(state, self.checkpoint_best)

    def load_checkpoint(self, file_name):
        if os.path.exists(file_name):
            checkpoint = torch.load(file_name)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.net.load_state_dict(checkpoint['net_state_dict'])
            self.optim_net.load_state_dict(checkpoint['net_optimizer'])
            self.manual_seed = checkpoint['seed']

            print("successfully load checkpoint")

        else:
            print("**First time to train**")

    def run(self):
        try:
            self.train()
            self.test()
        except KeyboardInterrupt:
            print("you have entered ctrl + c")

    def train(self):
        self.load_checkpoint(file_name = self.checkpoint_current)
        for epoch in range(self.current_epoch, self.config.max_epoch + 1):
            self.current_epoch = epoch

            self.epoch_train_loss = AverageMeter()
            self.epoch_val_loss = AverageMeter()

            self.train_one_epoch()

            if self.epoch_train_loss.val < self.val_best_loss:
                self.val()

            self.epoch_val_loss.reset()
            self.epoch_train_loss.reset()
        
   
    def train_one_epoch(self):
        print("start train")
        train_tqdm_batch=tqdm(self.train_dataloader.train_loader, total=self.train_dataloader.num_iterations)

        self.net.train()

        accuracy = 0

        for idx, (inputs, labels, _) in enumerate(train_tqdm_batch):
            if self.cuda:
                inputs = inputs.cuda() #b, c, w, h
                labels = labels.float().cuda() #b, l

            self.optim_net.zero_grad()

            outputs = self.net(inputs)

            #loss
            loss = self.loss(outputs, labels)

            self.epoch_train_loss.update(loss)

            loss.backward()
            self.optim_net.step()

            argout = outputs.argmax(dim=-1)
            correct = (argout == labels.argmax(dim=-1))
            accuracy = correct.sum().float() / float(correct.size(0))

            train_tqdm_batch.set_description("epochs={:5d}, loss={:0.4f}, acc={:2.4f}".format(self.current_epoch, loss, accuracy))

            self.current_iteration += 1

        scheduler_lr = 0
        for param_group in self.optim_net.param_groups:
            scheduler_lr = param_group['lr']

        #log
        if os.path.exists(self.log_path_train) == False:
            with open(self.log_path_train, 'w', newline='') as train_writer_csv:
                header_list = ['epoch', 'loss', 'scheduler_lr', 'acc']
                train_writer = csv.DictWriter(train_writer_csv, fieldnames= header_list)
                train_writer.writeheader
        with open(self.log_path_train, 'a', newline='') as train_writer_csv:
            train_writer = csv.writer(train_writer_csv)
            train_writer.writerow([self.current_epoch, str(self.epoch_train_loss.val), str(scheduler_lr), ])
 
        self.save_checkpoint(current = True)
        
        train_tqdm_batch.close()

    def val(self):
        print("start val")
        self.load_checkpoint(file_name = self.checkpoint_current)

        self.net.eval()

        val_tqdm_batch = tqdm(self.val_dataloader.val_loader,
            total = self.val_dataloader.num_iterations)

        val_acc = 0

        #validation
        for val_idx, (inputs, labels, _) in enumerate(val_tqdm_batch):

            if self.cuda:
                inputs = inputs.cuda()
                labels = labels.float().cuda()
            
            with torch.no_grad():
                outputs = self.net(inputs)
            
            loss = self.loss(outputs, labels)
            self.epoch_val_loss.update(loss)

            argout = outputs.argmax(dim=-1)
            correct = (argout == labels.argmax(dim=-1))
            val_acc = correct.sum().float() / float(correct.size(0))

            val_tqdm_batch.set_description("epochs={:5d}, loss={:0.4f}, acc={:2.4f}".format(self.current_epoch, self.epoch_val_loss.val, val_acc))

        #log
        if os.path.exists(self.log_path_val) == False:
            with open(self.log_path_val, 'w', newline='') as val_writer_csv:
                header_list = ['epoch', 'loss','acc']
                val_writer = csv.DictWriter(val_writer_csv, fieldnames= header_list)
                val_writer.writeheader
        with open(self.log_path_val, 'a', newline='') as val_writer_csv:
            val_writer = csv.writer(val_writer_csv)
            val_writer.writerow([self.current_epoch, str(self.epoch_val_loss.val), val_acc.float()])

        
        k = self.epoch_val_loss.val
        self.scheduler.step(k.float())

        #save checkpoint
        if self.val_best_loss > self.epoch_val_loss.val :
            self.val_best_loss = self.epoch_val_loss.val        
            self.save_checkpoint(current = False)

        val_tqdm_batch.close()
    
    def test(self):
        print("start test")
        self.load_checkpoint(file_name = self.checkpoint_current)
        self.net.eval()

        test_tqdm_batch = tqdm(self.test_dataloader.test_loader,
            total = self.test_dataloader.num_iterations,
            desc="epoch={}-".format(self.current_epoch))

        # create empty dfs
        pred_df = pd.DataFrame(columns=["Image_Index"])
        true_df = pd.DataFrame(columns=["Image_Index"])

        #test
        for test_idx, (inputs, labels, _) in enumerate(test_tqdm_batch):
            if self.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            true_labels = labels.cpu().data.numpy()
            batch_size = true_labels.shape[0]

            outputs = self.net(inputs)
            outputs = nn.Sigmoid(outputs)
            probs = outputs.cpu().data.numpy()

            for j in range(0, batch_size):
                thisrow = {}
                truerow = {}
                thisrow["Image_Index"] = \
                    self.test_dataloader.transformed_datasets_.df.index[batch_size * test_idx + j]
                truerow["Image_Index"] = \
                    self.test_dataloader.transformed_datasets_.df.index[batch_size * test_idx + j]

                for k in range(len(self.test_dataloader.transformed_datasets_.PRED_LABEL)):
                    thisrow["prob_" + self.test_dataloader.transformed_datasets_.PRED_LABEL[k]] = probs[j, k]
                    truerow[self.test_dataloader.transformed_datasets_.PRED_LABEL[k]] = true_labels[j, k]

                pred_df = pred_df.append(thisrow, ignore_index=True)
                true_df = true_df.append(truerow, ignore_index=True)


        auc_df = pd.DataFrame(columns=["label", "auc"])

        # calc AUCs
        for column in true_df:
            if column == 'Image_Index':
                con
            actual = true_df[column]
            pred = pred_df["prob_" + column]
            thisrow = {}
            thisrow['label'] = column
            thisrow['auc'] = np.nan
            try:
                thisrow['auc'] = sklm.roc_auc_score(
                actual.to_numpy().astype(int), pred.to_numpy())
            except BaseException:
                print("can't calculate auc for " + str(column))
            auc_df = auc_df.append(thisrow, ignore_index=True)

        pred_df.to_csv(self.result_pred_path, index=False)
        auc_df.to_csv(self.result_auc_path, index=False)
            
        test_tqdm_batch.close()

    def visual(self):
        print("start visual")
        self.load_checkpoint(file_name = self.checkpoint_current)
        self.net.eval()
        # print(self.net)
        grad_cam = GradCam(self.net, target_layer=31)



        # print(self.input_path)
        # file_names = sorted(os.listdir(self.input_path))
        file_names = glob.glob(self.input_path + '/*.png')
        # print(file_names)
        # exit()
        eval_transform = transforms.Compose([
                    transforms.Resize(512),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        gen_count = 0
        for idx, (img, label, imgname) in enumerate(self.test_dataloader.test_loader):
            
            for j in range(len(imgname)):
                # file_name = file_name_.split('/')[-1]
                file_path = self.input_path + '/' + imgname[j]
                save_path = self.output_path + '/' + imgname[j]
                make_dir(save_path)

                img_pil = Image.open(file_path).convert('RGB')
                img_pil_resize = img_pil.resize((512,512))
                img_tensor = eval_transform(img_pil)
                
                #file name
                label = label[0]
                label = label.tolist()
                print(self.num_classes)
                print(label)


                print(len(label))
                print(len(self.num_classes))
                
                #original image name
                gt_name = ''
                names = zip(self.num_classes, label)
                for name, key in names:
                    if key == 1:
                        gt_name += str(name)
                        gt_name += '_'
                print(save_path + '/' + gt_name + '.png') #origin_image_path
                
                img_pil.save(save_path + '/' + gt_name + '.png')

                cam_list, preds_list = grad_cam.generate_cam(img_tensor)
                print(preds_list)
                
                for k in range(len(preds_list)):
                        

                        
                    _, heatmap_on_img = apply_colormap(img_pil_resize, cam_list[k])


                    # heatmap.save(self.output_image_cam_path + str(np.argmax(preds)) + "_" + file_name)
                    print(save_path + '/' + self.num_classes[k] + "_" + str(preds_list[k]) + ".png")
     
                    heatmap_on_img.save(save_path + '/' + self.num_classes[k] + "_" + str(preds_list[k]) + ".png")


                #save pred 
                if os.path.exists(self.output_pred_path + 'preds.csv') == False:
                    with open(self.output_pred_path + 'preds.csv', 'w', newline='') as pred_writer_csv:
                        header_list = ['file_name', 'pred']
                        pred_writer = csv.DictWriter(pred_writer_csv, fieldnames= header_list)
                        pred_writer.writeheader()
                with open(self.output_pred_path + 'preds.csv', 'a', newline='') as pred_writer_csv:
                    pred_writer = csv.writer(pred_writer_csv)
                    pred_writer.writerow([imgname, \
                            str(preds_list)])

                    gen_count +=1

                    print(self.generate_num, gen_count)

                    if self.generate_num == gen_count: 
                        print("end")
                        exit()

            


