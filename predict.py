import argparse
import os
import sys
import time
import shutil
import warnings

import numpy as np
import torch
import torch.nn as nn
# from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import DataLoader

from geo.data import CIFData, get_train_val_test_loader
from geo.data import collate_pool
from geo.model import geo_CGNN

parser = argparse.ArgumentParser(
    description='Geometric Information Enhanced Crystal Graph Network')
parser.add_argument('cifpath', help='path to the directory of CIF files.')
parser.add_argument('--modelpath', default='./experiments/models/AI4MATXOR.pth.tar', help='path to the trained model.')
#parser.add_argument('--configpath', default='./experiments/configs/config_onehot.json',help='path to the directory of config_one_hot.json file.')
parser.add_argument('--storage', choices=['file', 'ceph'],
                    default='file', help='Using file or '
                    'ceph storage (default: file)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--disable-save-torch', action='store_true',
                    help='Do not save CIF PyTorch data as .pkl; files')
parser.add_argument('--train-val-test', action='store_true',
                    help='Return training/validation/testing results')
parser.add_argument('--only-pred', action='store_true',
                    help='Jump Loss and MAE calcs')

args = parser.parse_args(sys.argv[1:])
if os.path.isfile(args.modelpath):
    print("=> loading model params '{}'".format(args.modelpath))
    model_checkpoint = torch.load(args.modelpath,
                                  map_location=lambda storage, loc: storage)
    model_args = argparse.Namespace(**model_checkpoint['args'])
    print("=> loaded model params '{}'".format(args.modelpath))
else:
    print("=> no model params found at '{}'".format(args.modelpath))

args.cuda = not args.disable_cuda and torch.cuda.is_available()
args.inference = True

best_mae_error = 1e10



def main():
    global args, model_args, best_mae_error

    # load data
    dataset = CIFData(args.cifpath, disable_save_torch=args.disable_save_torch, storage=args.storage)
    collate_fn = collate_pool

    if args.train_val_test:
        train_loader, val_loader, test_loader = get_train_val_test_loader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=model_args.batch_size,
            train_ratio=model_args.train_ratio,
            num_workers=args.workers,
            val_ratio=model_args.val_ratio,
            test_ratio=model_args.test_ratio,
            pin_memory=args.cuda,
            train_size=model_args.train_size,
            val_size=model_args.val_size,
            test_size=model_args.test_size,
            return_test=True)
    else:
        test_loader = DataLoader(dataset, batch_size=model_args.batch_size, shuffle=False,
                                 num_workers=args.workers, collate_fn=collate_fn,
                                 pin_memory=args.cuda)

    # make output folder if needed
    if not os.path.exists('output_pred'):
        os.mkdir('output_pred')

    # make and clean torch files if needed
    #torch_data_path = os.path.join(args.data_options[0], 'cifdata')
    # if args.clean_torch :
    #     dataset.clean_torch()
    #     shutil.rmtree(torch_data_path)
    # if os.path.exists(torch_data_path):
    #if not args.clean_torch:
    #        warnings.warn('Found cifdata folder at ' +
    #                      torch_data_path+'. Will read in .pkls as-available')
    # else:
        # os.mkdir(torch_data_path)

    # build model
    structures, _, _ = dataset[0]
    n_node_fea = 84
    #orig_atom_fea_len = structures[0].shape[-1]
    #nbr_fea_len = structures[1].shape[-1]
    model = geo_CGNN(n_node_fea)
    
    if args.cuda:
        model.cuda()

    # define loss func and optimizer
    criterion = nn.MSELoss()

    normalizer = Normalizer(torch.zeros(3))

    # optionally resume from a checkpoint
    
    print("=> loading model '{}'".format(args.modelpath))
    checkpoint = torch.load(args.modelpath,
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    normalizer.load_state_dict(checkpoint['normalizer'])
    print("=> loaded model '{}' (epoch {}, validation {})"
          .format(args.modelpath, checkpoint['epoch'],
                  checkpoint['best_mae_error']))
    

    if args.train_val_test:
        print('---------Evaluate Model on Train Set---------------')
        validate(train_loader, model, criterion, normalizer, test=True,
                 csv_name='train_results.csv')
        print('---------Evaluate Model on Val Set---------------')
        validate(val_loader, model, criterion, normalizer, test=True,
                 csv_name='val_results.csv')
        print('---------Evaluate Model on Test Set---------------')
        validate(test_loader, model, criterion, normalizer, test=True,
                 csv_name='test_results.csv')
    else:
        print('---------Evaluate Model on Dataset---------------')
        validate(test_loader, model, criterion, normalizer, test=True,
                 csv_name='predictions.csv')


def validate(val_loader, model, criterion, normalizer, test=False,
             csv_name='test_results.csv'):
    batch_time = AverageMeter()
    losses = AverageMeter()
    if True:
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input_, target, batch_cif_ids) in enumerate(val_loader):
        with torch.no_grad():
            if args.cuda:
                input_var = (Variable(input_.nodes.cuda(non_blocking=True)),
                             Variable(input_.edge_sources.cuda(non_blocking=True)),
                             Variable(input_.edge_targets.cuda(non_blocking=True)),
                             Variable(input_.edge_distance.cuda(non_blocking=True)),
                             Variable(input_.graph_indices.cuda(non_blocking=True)),
                             Variable(input_.node_counts.cuda(non_blocking=True)),
                             Variable(input_.combine_sets.cuda(non_blocking=True)),
                             Variable(input_.plane_wave.cuda(non_blocking=True)))
            else:
                input_var = (Variable(input_.nodes),
                             Variable(input_.edge_sources),
                             Variable(input_.edge_targets),
                             Variable(input_.edge_distance),
                             Variable(input_.graph_indices),
                             Variable(input_.node_counts),
                             Variable(input_.combine_sets),
                             Variable(input_.plane_wave))
        target_normed = normalizer.norm(target)
        with torch.no_grad():
            if args.cuda:
                target_var = Variable(target_normed.cuda(non_blocking=True))
            else:
                target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        if not args.only_pred:
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            if True:
                mae_error = mae(normalizer.denorm(output.data.cpu()), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                mae_errors.update(mae_error, target.size(0))
                if test:
                    test_pred = normalizer.denorm(output.data.cpu())
                    test_target = target
                    test_preds += test_pred.view(-1).tolist()
                    test_targets += test_target.view(-1).tolist()
                    test_cif_ids += batch_cif_ids
            else:
                accuracy, precision, recall, fscore, auc_score =\
                    class_eval(output.data.cpu(), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                accuracies.update(accuracy, target.size(0))
                precisions.update(precision, target.size(0))
                recalls.update(recall, target.size(0))
                fscores.update(fscore, target.size(0))
                auc_scores.update(auc_score, target.size(0))
                if test:
                    test_pred = torch.exp(output.data.cpu())
                    test_target = target
                    assert test_pred.shape[1] == 2
                    test_preds += test_pred[:, 1].tolist()
                    test_targets += test_target.view(-1).tolist()
                    test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if not args.only_pred:
                if True:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                              i+1, len(val_loader), batch_time=batch_time, loss=losses,
                              mae_errors=mae_errors))
                else:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                          'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                          'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                          'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                          'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                              i+1, len(val_loader), batch_time=batch_time, loss=losses,
                              accu=accuracies, prec=precisions, recall=recalls,
                              f1=fscores, auc=auc_scores))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                          i+1, len(val_loader), batch_time=batch_time))
                

    if test:
        star_label = '**'
        import csv
        with open(os.path.join('output_pred', csv_name), 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
        
    if not args.only_pred:
        if True:
            print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
                                                            mae_errors=mae_errors))
            return mae_errors.avg
        else:
            print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
                                                     auc=auc_scores))
            return auc_scores.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    """
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score
    """
    return 0


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
