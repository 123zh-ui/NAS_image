import sys
import logging
import time
import os
import torchvision.transforms as transforms
import torch
import numpy as np
import torchvision.datasets as datasets
import torch.utils.data as data

from search import *
from recovery import *
from utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

__all__ = ["Trainer"]


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

def onnx(model,onnx_name='image_nas8'):
    x = torch.ones(256, 3, 32, 32).to(device)
    #model = model.eval()
    model_trace = torch.jit.trace(model.to(device), x.cuda(),strict=False)
    #model_trace= torch.jit.script(model.to(device), x.cuda())
    torch.onnx.export(model_trace, # 搭建的网络
                            x.cuda(), # 输入张量
                            'ista_nas_single/onnx/'+onnx_name+".onnx", # 输出模型名称
                            opset_version=13,
                            input_names=["input"], # 输入命名
                            output_names=["output"], # 输出命名
                            dynamic_axes={'input':{0:'batch'}, 'output':{0:'batch'}}  # 动态轴
                            )


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.num_ops      = len(PRIMITIVES)
        self.proj_dims    = cfg.proj_dims
        self.sparseness   = cfg.sparseness
        self.steps        = cfg.steps

        self.search_trainer = InnerTrainer(cfg)
        self.num_edges = self.search_trainer.model.num_edges
        #self.train_queue, self.test_queue = self.set_dataloader()
        self.train_queue, self.test_queue = self.set_dataloader_imagenet()

    def set_dataloader(self):
        kwargs = {"num_workers": 2, "pin_memory": True}

        train_transform, valid_transform = cifar10_transforms(self.cfg)
        train_data = datasets.CIFAR10(
            root=self.cfg.data, train=True, download=True, transform=train_transform)
        test_data = datasets.CIFAR10(
            root=self.cfg.data, train=False, download=True, transform=valid_transform)

        train_queue = data.DataLoader(
            train_data, batch_size=self.cfg.batch_size,
            # sampler=data.sampler.SubsetRandomSampler(indices[:split]),
            shuffle=True, **kwargs)

        test_queue = data.DataLoader(
            test_data, batch_size=self.cfg.batch_size,
            shuffle=False, **kwargs)

        return train_queue, test_queue
    
    def set_dataloader_imagenet(self):
        data_dir = os.path.join(self.cfg.tmp_data_dir, 'images')
        traindir = os.path.join(data_dir, 'train')
        validdir = os.path.join(data_dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_data = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]))
        valid_data = datasets.ImageFolder(
            validdir,
            transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            normalize,
        ]))
 
        train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=self.cfg.batch_size, shuffle=True, pin_memory=True, num_workers=self.cfg.workers)

        valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=self.cfg.batch_size, shuffle=False, pin_memory=True, num_workers=self.cfg.workers)
        return train_queue, valid_queue

    def do_recovery(self, As, alpha, x_last=None, freeze_flag=None):
        xs = []
        for i in range(self.steps):
            if freeze_flag is not None and freeze_flag[i]:
                xs.append(x_last[i])#如果相邻两个的结构系数中，对应node的结构系数之间的差距足够小，就保留不对它进行修改
                continue
            lasso = LASSO(As[i].cpu().numpy().copy())
            b = alpha[i]
            x = lasso.solve(b)
            xs.append(x)

        return xs

    def do_search(self, A_normal, normal_biases, normal_freeze_flag,
                       A_reduce, reduce_biases, reduce_freeze_flag, epoch, all_freeze,alpha_list):
        if not all_freeze:
            self.search_trainer.model.init_proj_mat(A_normal, A_reduce)#这个只是简单的赋值
            self.search_trainer.model.freeze_alpha(normal_freeze_flag, reduce_freeze_flag)#少数情况会进行更新
            self.search_trainer.model.init_bias(normal_biases, reduce_biases)#这个也只是赋值
        # train
        train_acc, train_obj, alpha_list = self.search_trainer.train_epoch(
            self.train_queue, epoch, all_freeze, alpha_list)
        logging.info("train_acc {:.4f}".format(train_acc))
        #print(self.search_trainer.model.alphas_normal)
        #print(self.search_trainer.model.alphas_reduce)
        # valid
        valid_acc, valid_obj = self.search_trainer.validate(self.test_queue, all_freeze)
        logging.info("valid_acc {:.4f}".format(valid_acc))

        if not all_freeze:
            alphas = self.search_trainer.model.arch_parameters()
            alpha_normal = alphas[0].detach().cpu().numpy()
            alpha_reduce = alphas[1].detach().cpu().numpy()
            return alpha_normal, alpha_reduce, alpha_list
        return alpha_list
    #与two的一样的
    def sample_and_proj(self, base_As, xs):
        As= []
        biases = []
        for i in range(self.steps):
            A = base_As[i].numpy().copy()
            E = A.T.dot(A) - np.eye(A.shape[1])
            x = xs[i].copy()
            zero_idx = np.abs(x).argsort()[:-self.sparseness]
            x[zero_idx] = 0.
            A[:, zero_idx] = 0.
            As.append(torch.from_numpy(A).float())
            E[:, zero_idx] = 0.
            bias = E.T.dot(x).reshape(-1, self.num_ops)
            biases.append(torch.from_numpy(bias).float())

        biases = torch.cat(biases)

        return As, biases

    def show_selected(self, epoch, x_normals_last, x_reduces_last,
                                   x_normals_new, x_reduces_new):
        print("\n[Epoch {}]".format(epoch if epoch > 0 else 'initial'))
        # print("x_normals:\n", x_normals)
        # print("x_reduces:\n", x_reduces)
        print("x_normals distance:")
        normal_freeze_flag = []
        reduce_freeze_flag = []
        for i, (x_n_b, x_n_a) in enumerate(zip(x_normals_last, x_normals_new)):
            dist = np.linalg.norm(x_n_a - x_n_b, 2)#求解前后两次训练的结构系数之间的二范数
            normal_freeze_flag.append(False if epoch == 0 else dist <= 1e-3)
            print("Step {}: L2 dist is {}. {}".format(i+1, dist,
                            "freeze!!!" if normal_freeze_flag[-1] else "active"))
        print("x_reduces distance:")
        for i, (x_r_b, x_r_a) in enumerate(zip(x_reduces_last, x_reduces_new)):
            dist = np.linalg.norm(x_r_a - x_r_b, 2)
            reduce_freeze_flag.append(False if epoch == 0 else dist <= 1e-3)
            print("Step {}: L2 dist is {}. {}".format(i+1, dist,
                            "freeze!!!" if reduce_freeze_flag[-1] else "active"))

        print("normal cell:")
        gene_normal = []
        for i, x in enumerate(x_normals_new):
            id1, id2 = np.abs(x).argsort()[-2:]
            print("Step {}: node{} op{}, node{} op{}".format(
                i + 1, id1 // self.num_ops,
                       id1 % self.num_ops,
                       id2 // self.num_ops,
                       id2 % self.num_ops))
            gene_normal.append((PRIMITIVES[id1 % self.num_ops], id1 // self.num_ops))
            gene_normal.append((PRIMITIVES[id2 % self.num_ops], id2 // self.num_ops))

        print("reduction cell:")
        gene_reduce = []
        for i, x in enumerate(x_reduces_new):
            id1, id2 = np.abs(x).argsort()[-2:]
            print("Step {}: node{} op{}, node{} op{}".format(
                i + 1, id1 // self.num_ops,
                       id1 % self.num_ops,
                       id2 // self.num_ops,
                       id2 % self.num_ops))
            gene_reduce.append((PRIMITIVES[id1 % self.num_ops], id1 // self.num_ops))
            gene_reduce.append((PRIMITIVES[id2 % self.num_ops], id2 // self.num_ops))

        concat = range(2, 2 + len(x_normals_new))
        genotype = Genotype(
            normal = gene_normal, normal_concat = concat,
            reduce = gene_reduce, reduce_concat = concat)
        print(genotype)#下面这三行就是新加的
        model_cifar = NetworkCIFAR(36, 10, 20, True, genotype)#与two-stage不一样
        param_size = count_parameters_in_MB(model_cifar)
        logging.info('param size = %fMB', param_size)

        return normal_freeze_flag, reduce_freeze_flag, param_size

    def alpha_deal(self, alpha_list):
        alpha_len = len(alpha_list)
        alpha_array = np.ones((alpha_len*8, 6), float)
        for j, alpha in enumerate(alpha_list):
            for i in range(2):
                a = alpha[i] != 0
                c = alpha[i]
                b = np.nonzero(a)
                alpha_array[j*8:(j+1)*8,i*3:(i+1)*3-1] = b.cpu().numpy()
                temp = np.array(c[a].cpu().detach().numpy())
                alpha_array[j*8:(j+1)*8,(i+1)*3-1:(i+1)*3] = temp.reshape((8,1))
        np.savetxt( "alpha_list1_20cell.csv", alpha_array, delimiter="," )
        
    def train(self):
        alpha_list = list()
        time_start = time.time()
        base_A_normals = []
        base_A_reduces = []#在onestage之中，每个cell的结构没有改变，改变的是cell的层数
        #14，21，28，35
        for i in range(self.steps):
            base_A_normals.append(
                torch.from_numpy(np.random.rand(self.proj_dims, (i+2) * self.num_ops)))#2*7，3*7，4*7，5*7
            base_A_reduces.append(
                torch.from_numpy(np.random.rand(self.proj_dims, (i+2) * self.num_ops)))
        #这里x_noemals_new表示的是z，
        alpha_normal = self.search_trainer.model.alphas_normal_.detach().cpu().numpy()
        alpha_reduce = self.search_trainer.model.alphas_reduce_.detach().cpu().numpy()
        x_normals_new = self.do_recovery(base_A_normals, alpha_normal)
        x_reduces_new = self.do_recovery(base_A_reduces, alpha_reduce)
        #torch.stack() 官方解释：沿着一个新维度对输入张量序列进行连接
        x_normals_last = x_normals_new.copy()
        x_reduces_last = x_reduces_new.copy()#dist相邻两次结构系数的差值
        #freeze指的当epoch==0，freeze=false，当epoch！=0时，dist《 1e-03时，freeze=True
        normal_freeze_flag, reduce_freeze_flag, _ = self.show_selected(
            0, x_normals_last, x_reduces_last, x_normals_new, x_reduces_new)


        for i in range(self.cfg.epochs):
            A_normals, normal_biases = self.sample_and_proj(
                base_A_normals, x_normals_last)#和two-stage的一样
            A_reduces, reduce_biases = self.sample_and_proj(
                base_A_reduces, x_reduces_last)
            print("\nDoing Search ...")
            self.search_trainer.model.drop_path_prob = 0 #self.cfg.drop_path_prob * i / self.cfg.epochs
            alpha_normal, alpha_reduce, alpha_list = self.do_search(
                A_normals, normal_biases, normal_freeze_flag,
                A_reduces, reduce_biases, reduce_freeze_flag, i+1, False,alpha_list)#这个地方all_freeze最开始是False
            
            for i in range(4):
               x_normals_last[i] 
            if False not in normal_freeze_flag and False not in reduce_freeze_flag:
                self.alpha_deal(alpha_list)
                time_end = time.time()
                logging.info('收敛时间 = %fs', time_end - time_start)
                print('\n收敛的时间',time_end - time_start)
                break
            print("Doing Recovery ...")
            x_normals_new = self.do_recovery(base_A_normals, alpha_normal,
                    x_normals_last, normal_freeze_flag)
            x_reduces_new = self.do_recovery(base_A_reduces, alpha_reduce,
                    x_reduces_last, reduce_freeze_flag)
            ## update freeze flag
            normal_freeze_flag, reduce_freeze_flag, param_size = self.show_selected(
                i+1, x_normals_last, x_reduces_last, x_normals_new, x_reduces_new)
            if param_size >= 3.7: # large model may cause out of memory !!!
                print('-------------> rejected !!!')#上面那一步返回一轮更新后，网络的变化，如果满足条件
                continue
            x_normals_last = x_normals_new
            x_reduces_last = x_reduces_new

        print("\n --- Architecture Fixed, Retrain for {} Epochs --- \n".format(self.cfg.epochs))
        for i in range(self.cfg.epochs):
            self.search_trainer.model.drop_path_prob = self.cfg.drop_path_prob * i / self.cfg.epochs
            alpha_list = self.do_search(
                A_normals, normal_biases, normal_freeze_flag,
                A_reduces, reduce_biases, reduce_freeze_flag, i+1, True,alpha_list)
        #print(alpha_list)
        #self.alpha_deal(alpha_list)
        onnx(self.search_trainer.model,"targetnet_image")
        #for i, alpha in enumerate(alpha_list):
            
#BN的作用
# 原因在于神经网络学习过程本质就是为了学习数据分布，
# 一旦训练数据与测试数据的分布不同，那么网络的泛化能力也大大降低；
# 另外一方面，一旦每批训练数据的分布各不相同(batch 梯度下降)，
# 那么网络就要在每次迭代都去学习适应不同的分布，
# 这样将会大大降低网络的训练速度，这也正是为什么我们需要对数据都要做一个归一化预处理的原因。
#