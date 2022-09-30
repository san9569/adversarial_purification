import os
import math
import json
import random
import wandb
import torch
import losses
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
import torchvision.transforms as transforms
import torchattacks

import configargparse

from tqdm import tqdm

# Data
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from dataloader import get_dataloader

from autoattack import AutoAttack
from torch.autograd import Variable
from collections import OrderedDict
from torchvision.utils import save_image
from utils import *


# Classifier
from models.cifar10.resnet import resnet50

parser = configargparse.ArgumentParser()

# Training parameter
parser.add_argument('--config', is_config_file=True, help='config file path')
parser.add_argument("--project_name", type=str, default="white-box", help="project name for wandb")
parser.add_argument("--exp_name", type=str, default="base", help="Experiment name for wandb")

parser.add_argument("--no_train", action="store_true")
parser.add_argument("--epochs", type=int, default=5, help="number of epochs of training")
parser.add_argument('--bsz', type=int, default=32)
parser.add_argument('--test_bsz', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument("--gpu", type=int, default=0, help="GPU ID")

parser.add_argument('--dataset', type=str, default="cifar10", choices=["imagenet", "cifar10"], metavar='', help='Dataset')
parser.add_argument('--dataset_path', type=str, default='/shared/public/dataset/ILSVRC2012', metavar='', help='Dataset path')
parser.add_argument("--train_subsample", type=float, default=0.08, help="Subsample ratio for sampling the train dataset")
parser.add_argument("--test_subsample", type=float, default=0.025, help="Subsample ratio for sampling the test dataset")

parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
parser.add_argument("--decay_step", type=float, default=1250)
parser.add_argument("--decay", type=float, default=0.8, help="Weight decay")

# Log argument
parser.add_argument('--print_freq', type=int, default=20, metavar='', help='Frequency to print training process')
parser.add_argument("--sample_interval", type=int, default=100, help="Sample download ")
parser.add_argument('--save_freq', type=int, default=1000, metavar='', help='Frequency to print training process')

# Result Path
parser.add_argument('--result_path', type=str, default='./result', metavar='', help='Random seed')

# Checkpoints
parser.add_argument('--resume', action="store_true")

# Attack
parser.add_argument("--attack", type=str, default="pgd", choices=["pgd", "fgsm","aa", "cw"], help="Attack method")
parser.add_argument("--eval_attack", type=str, default="pgd", choices=["pgd", "fgsm","aa", "cw"], help="Attack method")
parser.add_argument("--pgd_step", type=int, default=20)
parser.add_argument("--eval_pgd_step", type=int, default=40)
parser.add_argument("--eot_iter", type=int, default=10)
parser.add_argument("--eps", type=float, default=8/255, help="attack epilson")

# Loss function
parser.add_argument("--l1_coeff", type=float, default=1.0, help="Coefficient of L1 loss")
parser.add_argument("--kl_coeff", type=float, default=0.01)

parser.add_argument("--classifier", type=str, default="resnet50", choices=["resnet50", "wrnet-28-10", "wrnet-34-10", "wrnet-70-16"], help="Backbone network for classifier")
parser.add_argument("--eval_classifier", type=str, default="resnet50", choices=["resnet50", "wrnet-28-10", "wrnet-34-10", "wrnet-70-16"], help="Backbone network for classifier")
parser.add_argument("--pur_depth", type=int, default=12, help="Depth of purifier block")

args = parser.parse_args()
print(args)

# GPU Setting
os.environ["CUDA_DEVICE_ORDER"] ="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu

device = 'cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu'

import gc
gc.collect()
torch.cuda.empty_cache()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed) # if use multi-GPU
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ["PYTHONHASHSEED"] = str(args.seed)

# Result path
if not os.path.exists(args.result_path):
    os.makedirs(args.result_path)
    
result_path = os.path.join(args.result_path, args.project_name, args.exp_name)
if not os.path.exists(result_path):
    os.makedirs(result_path)

import logging
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

# Load pre-trained classifier
norm = True
if args.dataset == "imagenet":
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    from models.imagenet.resnet_ import resnet50
    classifier = resnet50(pretrained=True, norm=norm).to(device)
else:
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2471, 0.2435, 0.2616]
    from models.cifar10.resnet import resnet50
    from models.cifar10.wide_resnet import wrnet_28_10, wrnet_34_10, wrnet_70_16
    if args.classifier == "resnet50":
        classifier = resnet50(pretrained=True, device=device, norm=norm).to(device)
    elif args.classifier == "wrnet-28-10":
        classifier = wrnet_28_10(pretrained=True, device=device, norm=norm).to(device)
    elif args.classifier == "wrnet-34-10":
        classifier = wrnet_34_10(pretrained=True, device=device, norm=norm).to(device)
    elif args.classifier == "wrnet-70-16":
        classifier = wrnet_70_16(pretrained=True, device=device, norm=norm).to(device)
classifier.eval()

from models.temp_snn import PAP, Purifier_Classifier
purifier = PAP()

purifier = purifier.to(device)
model = Purifier_Classifier(purifier, classifier).to(device)

# Attack
if args.attack == "pgd":
    attack = torchattacks.PGD(model, eps=args.eps, alpha=2/255, steps=args.pgd_step)
    blackbox_attack = torchattacks.PGD(classifier, eps=args.eps, alpha=2/255, steps=args.pgd_step)
elif args.attack == "eot-pgd":
    attack = torchattacks.EOTPGD(model, eps=args.eps, alpha=2/255, steps=args.pgd_step, eot_iter=args.eot_iter, random_start=True)
    blackbox_attack = torchattacks.EOTPGD(classifier, eps=args.eps, alpha=2/255, steps=args.pgd_step, eot_iter=args.eot_iter, random_start=True)
elif args.attack == "fgsm":
    attack = torchattacks.FGSM(model, eps=args.eps)
    blackbox_attack = torchattacks.FGSM(classifier, eps=args.eps)
elif args.attack == "cw":
    attack = torchattacks.CW(model, c=1, kappa=0, steps=100, lr=0.01)
    blackbox_attack = torchattacks.CW(classifier, c=1, kappa=0, steps=100, lr=0.01)
elif args.attack == "aa":
    attack = torchattacks.AutoAttack(model, norm='Linf', eps=args.eps, version='standard', n_classes=10, seed=args.seed, verbose=False)
    blackbox_attack = torchattacks.AutoAttack(classifier, norm='Linf', eps=args.eps, version='standard', n_classes=10, seed=args.seed, verbose=False)
else:
    raise Exception("Attack method is wrong. It should be pgd or aa.")


if not args.no_train:
    os.environ['WANDB_API_KEY'] = "0ee23525f6f4ddbbab74086ddc0b2294c7793e80"
    wandb.init(project=args.project_name, entity="psj", name=args.exp_name)
    wandb.config.update(args)
    
    assert args.attack == args.eval_attack, "Attack methods at training and evaluation phase are different!"
    
    # Optimizers
    optimizer = torch.optim.Adam(model.purifier.parameters(), lr=args.lr, betas=(args.b1, args.b2), weight_decay=args.weight_decay)
        
    # Load the checkpoint
    if args.resume:
        checkpoint = torch.load(os.path.join(result_path, "%s_%s_%s_ckpt.pth" % (args.dataset, args.attack, args.classifier)))
        model.purifier.load_state_dict(checkpoint["purifier"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        logging.info("Load the model at step %d" % checkpoint["iteration"])
        start_iteration = checkpoint["iteration"]
    else:
        start_iteration = 0
        
    losses_dict = {}
    param_dict = OrderedDict()
    
    # Loss function
    perceptual_loss = losses.VGGPerceptualLoss(dataset=args.dataset).to(device)

    train_loader, data_norm = get_dataloader(dataset_name=args.dataset, which="train", subsample=args.train_subsample, args=args)
    test_loader, _ = get_dataloader(dataset_name=args.dataset, which="val", subsample=args.test_subsample, args=args)
    
    Tensor = torch.cuda.FloatTensor if device == 'cuda:0' else torch.FloatTensor
    
    total_iteration = args.epochs * len(train_loader)
    decay_step = args.decay_step * len(train_loader)
    lr = args.lr
    train_loader_iterator = iter(train_loader)
    
    for idx, iteration in enumerate(range(start_iteration, total_iteration)):
        try:
            (x, y) = next(train_loader_iterator)
        except StopIteration:
            np.random.seed()  # Ensure randomness
            # Some cleanup
            train_loader_iterator = None
            torch.cuda.empty_cache()
            gc.collect()
            train_loader_iterator = iter(train_loader)
            (x, y) = next(train_loader_iterator)
        
        for param in model.purifier.parameters():
            param.requires_grad = True

        x, y = x.to(device), y.to(device)
        model.purifier.train()
        
        adv = attack(x, y)
            
        pred_pur, x_pur, kl_loss = model(adv, True)
        
        if args.l1_coeff != 0:
            losses_dict["L1"] = abs(x - x_pur).mean()
            total_loss = losses_dict["L1"] * args.l1_coeff
            
        if args.kl_coeff != 0:
            losses_dict["KL"] = kl_loss
            total_loss += losses_dict["KL"] * args.kl_coeff
            
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        ## Calculate train accuracy
        with torch.no_grad():
            
            pred_cln, x_cln, _ = model(x, True)
            # Natural
            out = model.classifier(x)
            _, y_pred = torch.max(out.data, 1)
            correct_nat = (y_pred == y).sum().item()
            
            # Clean
            _, y_cln = torch.max(pred_cln.data, 1)
            correct_cln = (y_cln == y).sum().item()
            
            # Adversarial
            out = model.classifier(adv)
            _, y_pred = torch.max(out.data, 1)
            correct_adv = (y_pred == y).sum().item()
            
            # Robust
            _, y_rob = torch.max(pred_pur.data, 1)
            correct_rob = (y_rob == y).sum().item()
        
        losses_dict['nat_acc'] = (correct_nat/x.size(0)) * 100
        losses_dict['cln_acc'] = (correct_cln/x.size(0)) * 100
        losses_dict['adv_acc'] = (correct_adv/x.size(0)) * 100
        losses_dict['rob_acc'] = (correct_rob/x.size(0)) * 100
        
        wandb.log({"train_nat_acc":losses_dict['nat_acc'],
                  "train_cln_acc":losses_dict['cln_acc'],
                  "train_adv_acc":losses_dict['adv_acc'],
                  "train_rob_acc":losses_dict['rob_acc'],
                  "iteration":iteration})
        
        # Print the training process
        if iteration % args.print_freq == 0:
            logging.info("")
            logging.info("[Epoch %d/%d] [Iteration %d/%d]" % 
                         (iteration // (len(train_loader)), args.epochs, iteration, total_iteration))
            print_loss(losses_dict)

        wandb.log(losses_dict.copy())

        # Save the model every specified iteration.
        if iteration != 0 and iteration % args.save_freq == 0:
            ckpt_path = os.path.join(result_path, "%s_%s_%s_ckpt.pth" 
                                     % (args.dataset, args.attack, args.classifier))
            torch.save({"purifier":model.purifier.state_dict(),
                        "optimizer":optimizer.state_dict(),
                        "iteration":iteration}, ckpt_path)
            
        if iteration % args.sample_interval == 0:
            correct_test = 0
            correct_adv_test = 0
            correct = 0
            total_test = 0
            fig, axs = plt.subplots(4,4)
            for idx, (x_test, y_test) in enumerate(test_loader):
                x_test, y_test = x_test.to(device), y_test.to(device)

                adv_test = attack(x_test, y_test)
                    
                model.eval()
                
                with torch.no_grad():
                    pred_cln, x_cln, _ = model(x_test, True)
                    _, y_cln = torch.max(pred_cln.data, 1)
                    correct_test += (y_cln == y_test).sum().item()
                    
                    pred_pur, x_pur, _ = model(adv_test, True)
                    _, y_pur = torch.max(pred_pur.data, 1)
                    correct_adv_test += (y_pur == y_test).sum().item()
        
                    pred = model.classifier(x_test)
                    _, y_pred = torch.max(pred.data, 1)
                    correct += (y_pred == y_test).sum().item()

                total_test += x_test.size(0)

                if idx == 0:
                    axs[0, 0].set_title("Raw")
                    axs[0, 1].set_title("Clean")
                    axs[0, 2].set_title("Perturbed")
                    axs[0, 3].set_title("Purified")
                    for i in range(4):
                        axs[i, 0].imshow(recover_image(x_test[i], cv=False)),  axs[i, 0].axis('off')
                        axs[i, 1].imshow(recover_image(x_cln[i], cv=False)), axs[i, 1].axis('off')
                        axs[i, 2].imshow(recover_image(adv_test[i], cv=False)), axs[i, 2].axis('off')
                        axs[i, 3].imshow(recover_image(x_pur[i], cv=False)), axs[i, 3].axis('off')
            plt.subplots_adjust(wspace=-0.5, hspace=0.01)
            plt.savefig(os.path.join(result_path, "fig%d.png" % iteration), dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

            # clean_robust_acc = correct_test.avg * 100
            # pur_robust_acc = correct_adv_test.avg * 100
            nat_acc = (correct / total_test) * 100
            clean_robust_acc = (correct_test / total_test) * 100
            pur_robust_acc = (correct_adv_test / total_test) * 100

            logging.info("")
            logging.info(colors.YELLOW + "[Evaluation on test subset of %s]" % args.dataset + colors.WHITE)
            logging.info("Natural accuracy: %.4f" % nat_acc)
            logging.info("Clean accuracy: %.4f" % clean_robust_acc)
            logging.info("Robust accuracy: %.4f" % pur_robust_acc)
            wandb.log({"test_nat_acc": nat_acc})
            wandb.log({"test_cln_acc": clean_robust_acc})
            wandb.log({"test_rob_acc": pur_robust_acc})
            torch.cuda.empty_cache()

        torch.cuda.empty_cache()

    gc.collect()
    torch.cuda.empty_cache()
    logging.info("Saving the final model...")
    ckpt_path = os.path.join(result_path, "%s_%s_%s_final.pth" % (args.dataset, args.attack, args.classifier))
    torch.save({"purifier":purifier.state_dict(),
                "optimizer":optimizer.state_dict(),
                "iteration":iteration}, ckpt_path)
    logging.info("Training finished.")

else:
    logging.info("")
    logging.info("Skip training")
    checkpoint = torch.load(os.path.join(result_path, "%s_%s_%s_final.pth" % (args.dataset, args.attack, args.classifier)))
    purifier.load_state_dict(checkpoint["purifier"])
    logging.info("Load the model at step %d" % checkpoint["iteration"])
    
    # Load pre-trained classifier
    norm = True
    if args.dataset == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        from models.imagenet.resnet_ import resnet50
        classifier = resnet50(pretrained=True, norm=norm).to(device)
    else:
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2471, 0.2435, 0.2616]
        from models.cifar10.resnet import resnet50
        from models.cifar10.wide_resnet import wrnet_28_10, wrnet_34_10, wrnet_70_16
        if args.eval_classifier == "resnet50":
            classifier = resnet50(pretrained=True, device=device, norm=norm).to(device)
        elif args.eval_classifier == "wrnet-28-10":
            classifier = wrnet_28_10(pretrained=True, device=device, norm=norm).to(device)
        elif args.eval_classifier == "wrnet-34-10":
            classifier = wrnet_34_10(pretrained=True, device=device, norm=norm).to(device)
        elif args.eval_classifier == "wrnet-70-16":
            classifier = wrnet_70_16(pretrained=True, device=device, norm=norm).to(device)
    
    model = Purifier_Classifier(purifier, classifier).to(device)
    
    # Attack
    if args.eval_attack == "pgd":
        attack = torchattacks.PGD(model, eps=args.eps, alpha=2/255, steps=args.eval_pgd_step)
    elif args.attack == "eot-pgd":
        attack = torchattacks.EOTPGD(model, eps=args.eps, alpha=2/255, steps=args.pgd_step, eot_iter=args.eot_iter, random_start=True)
    elif args.eval_attack == "fgsm":
        attack = torchattacks.FGSM(model, eps=args.eps)
    elif args.eval_attack == "cw":
        attack = torchattacks.CW(model, c=1, kappa=0, steps=100, lr=0.01)
    elif args.eval_attack == "aa":
        attack = torchattacks.AutoAttack(model, norm='Linf', eps=args.eps, version='standard', n_classes=10, seed=args.seed, verbose=False)
    else:
        raise Exception("Attack method is wrong. It should be pgd or aa.")

logging.info("")
logging.info("Start final evaluation")
test_loader, _ = get_dataloader(dataset_name=args.dataset, which="val", subsample=1.0, args=args)
correct_test = 0
correct_adv_test = 0
correct = 0
correct_adv = 0
total_test = 0
correct_black_rob = 0
for idx, (x_test, y_test) in enumerate(tqdm(test_loader)):
    x_test, y_test = x_test.to(device), y_test.to(device)
    
    adv_test = attack(x_test, y_test)
    adv_unseen = blackbox_attack(x_test, y_test)
    
    model.eval()
    
    # Natural accuracy on clean image
    out = classifier(x_test)
    _, y_pred = torch.max(out.data, 1)
    correct += (y_pred == y_test).sum().item()
    
    # Natural accuracy on adversarial image
    out = classifier(adv_unseen)
    _, y_pred = torch.max(out.data, 1)
    correct_adv += (y_pred == y_test).sum().item()
    
    # Clean accuracy
    pred_cln = model(x_test)
    _, y_cln = torch.max(pred_cln.data, 1)
    correct_test += (y_cln == y_test).sum().item()
    
    # Robust accuracy
    """ White-box setting """
    pred_pur = model(adv_test)
    _, y_pur = torch.max(pred_pur.data, 1)
    correct_adv_test += (y_pur == y_test).sum().item()
    """ Black-box setting """
    out = model(adv_unseen)
    _, y_pred = torch.max(out.data, 1)
    correct_black_rob += (y_pred == y_test).sum().item()

    total_test += x_test.size(0)
        
logging.info("")
logging.info(colors.YELLOW + f"[Final evaluation on testset of {args.dataset}]" + colors.WHITE)
logging.info(f"Attack: {args.eval_attack}")
logging.info(f"Classifier: {args.eval_classifier}")
logging.info(f"Natural acc: {(correct / total_test) * 100}%")
logging.info(f"Attacked acc: {(correct_adv / total_test) * 100:.2f}%")
logging.info(f"Clean acc: {(correct_test / total_test) * 100:.2f}%")
logging.info(f"[White-box] Robust acc: {(correct_adv_test / total_test) * 100:.2f}%")
logging.info(f'[Gray-box] Robust acc: {(correct_black_rob / total_test) * 100:.2f}%')
torch.cuda.empty_cache()