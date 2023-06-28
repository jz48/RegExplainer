import torch
import numpy as np
from utils.training import GNNTrainer
from utils.explaining import MainExplainer

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

_explainer = 'GNNE'


def explain_(dataset, model, explainer_name):
    explainer = MainExplainer(dataset, model, explainer_name)
    auc, auc_std, rdd, rdd_std = explainer.explain()
    print(dataset, model, explainer_name, 'auc: ', auc, 'auc_std: ', auc_std, 'rdd: ', rdd, 'rdd_std: ', rdd_std)


def runs():
    explainers = ['GNNE', 'PGE', 'MIXGNNE', 'NAMGNNE', 'NAMPGE', 'MIXPGE']   # , 'CTLGNNE'
    # explainers = ['NAMGNNE']
    explainers = ['CTLPGE']
    models = ['GraphGCN']
    datasets = ['ba2motif', 'bareg1', 'bareg2']
    loss_type = 'ctl'
    for d in datasets:
        for e in explainers:
            for m in models:
                explainer = MainExplainer(d, m, e, wandb_log=False, loss_type=loss_type, force_run=True,
                                          save_explaining_results=False)
                auc, auc_std, rdd, rdd_std = explainer.explain()
                print(d, m, e, 'auc: ', auc, 'auc_std: ', auc_std, 'rdd: ', rdd, 'rdd_std: ',
                      rdd_std)


def run_crippen():
    explainers = ['GNNE', 'PGE', 'MIXGNNE', 'NAMGNNE', 'NAMPGE', 'MIXPGE']   # , 'CTLGNNE'
    # explainers = ['NAMGNNE']
    explainers = ['GNNE', 'PGE']
    models = ['GraphGCN']
    datasets = ['crippen']
    loss_type = 'ib'
    for d in datasets:
        for e in explainers:
            for m in models:
                explainer = MainExplainer(d, m, e, wandb_log=False, loss_type=loss_type, force_run=False,
                                          save_explaining_results=True)
                auc, auc_std, rdd, rdd_std = explainer.explain()
                print(d, m, e, 'auc: ', auc, 'auc_std: ', auc_std, 'rdd: ', rdd, 'rdd_std: ',
                      rdd_std)

    explainers = ['CTLMIXSFTPGE', 'CTLALTMIXPGE']
    models = ['GraphGCN']
    datasets = ['crippen', 'bareg1', 'bareg2']
    loss_type = 'ctl'
    for d in datasets:
        for e in explainers:
            for m in models:
                explainer = MainExplainer(d, m, e, wandb_log=False, loss_type=loss_type, force_run=True,
                                          save_explaining_results=True)
                auc, auc_std, rdd, rdd_std = explainer.explain()
                print(d, m, e, 'auc: ', auc, 'auc_std: ', auc_std, 'rdd: ', rdd, 'rdd_std: ',
                      rdd_std)


def run_triangles():
    explainers = ['GNNE', 'PGE', 'MIXGNNE', 'NAMGNNE', 'NAMPGE', 'MIXPGE']
    models = ['GraphGCN']
    datasets = ['triangles']
    loss_type = 'ib'
    for d in datasets:
        for e in explainers:
            for m in models:
                explainer = MainExplainer(d, m, e, wandb_log=False, loss_type=loss_type, force_run=False,
                                          save_explaining_results=True)
                auc, auc_std, rdd, rdd_std = explainer.explain()
                print(d, m, e, 'auc: ', auc, 'auc_std: ', auc_std, 'rdd: ', rdd, 'rdd_std: ',
                      rdd_std)

    explainers = ['CTLMIXSFTPGE', 'CTLALTMIXPGE', 'CTLPGE']
    models = ['GraphGCN']
    datasets = ['triangles']
    loss_type = 'ctl'
    for d in datasets:
        for e in explainers:
            for m in models:
                explainer = MainExplainer(d, m, e, wandb_log=False, loss_type=loss_type, force_run=False,
                                          save_explaining_results=True)
                auc, auc_std, rdd, rdd_std = explainer.explain()
                print(d, m, e, 'auc: ', auc, 'auc_std: ', auc_std, 'rdd: ', rdd, 'rdd_std: ',
                      rdd_std)

    explainers = ['CTLPGE']
    models = ['GraphGCN']
    datasets = ['triangles']
    loss_type = 'ctl'
    for d in datasets:
        for e in explainers:
            for m in models:
                explainer = MainExplainer(d, m, e, wandb_log=False, loss_type=loss_type, force_run=True,
                                          save_explaining_results=True)
                auc, auc_std, rdd, rdd_std = explainer.explain()
                print(d, m, e, 'auc: ', auc, 'auc_std: ', auc_std, 'rdd: ', rdd, 'rdd_std: ',
                      rdd_std)


def run_triangles_small():
    explainers = ['GNNE', 'PGE', 'MIXGNNE', 'NAMGNNE', 'NAMPGE', 'MIXPGE']
    models = ['GraphGCN']
    datasets = ['triangles_small']
    loss_type = 'ib'
    for d in datasets:
        for e in explainers:
            for m in models:
                explainer = MainExplainer(d, m, e, wandb_log=False, loss_type=loss_type, force_run=False,
                                          save_explaining_results=True)
                auc, auc_std, rdd, rdd_std = explainer.explain()
                print(d, m, e, 'auc: ', auc, 'auc_std: ', auc_std, 'rdd: ', rdd, 'rdd_std: ',
                      rdd_std)

    explainers = ['CTLMIXSFTPGE', 'CTLALTMIXPGE', 'CTLPGE']
    models = ['GraphGCN']
    datasets = ['triangles_small']
    loss_type = 'ctl'
    for d in datasets:
        for e in explainers:
            for m in models:
                explainer = MainExplainer(d, m, e, wandb_log=False, loss_type=loss_type, force_run=False,
                                          save_explaining_results=True)
                auc, auc_std, rdd, rdd_std = explainer.explain()
                print(d, m, e, 'auc: ', auc, 'auc_std: ', auc_std, 'rdd: ', rdd, 'rdd_std: ',
                      rdd_std)

    explainers = ['CTLPGE']
    models = ['GraphGCN']
    datasets = ['triangles']
    loss_type = 'ctl'
    for d in datasets:
        for e in explainers:
            for m in models:
                explainer = MainExplainer(d, m, e, wandb_log=False, loss_type=loss_type, force_run=True,
                                          save_explaining_results=True)
                auc, auc_std, rdd, rdd_std = explainer.explain()
                print(d, m, e, 'auc: ', auc, 'auc_std: ', auc_std, 'rdd: ', rdd, 'rdd_std: ',
                      rdd_std)


def run_baregs():
    force_run = False
    models = ['GraphGCN']
    datasets = ['bareg1', 'bareg2']

    explainers = ['GNNE', 'PGE', 'MIXGNNE', 'MIXPGE', 'NAMGNNE', 'NAMPGE']
    loss_type = 'ib'
    for d in datasets:
        for e in explainers:
            for m in models:
                explainer = MainExplainer(d, m, e, wandb_log=False, loss_type=loss_type, force_run=force_run,
                                          save_explaining_results=True)
                auc, auc_std, rdd, rdd_std = explainer.explain()
                print(d, m, e, 'auc: ', auc, 'auc_std: ', auc_std, 'rdd: ', rdd, 'rdd_std: ',
                      rdd_std)

    explainers = ['CTLPGE', 'CTLMIXSFTPGE', 'CTLALTMIXPGE']
    loss_type = 'ctl'
    for d in datasets:
        for e in explainers:
            for m in models:
                explainer = MainExplainer(d, m, e, wandb_log=False, loss_type=loss_type, force_run=force_run,
                                          save_explaining_results=True)
                auc, auc_std, rdd, rdd_std = explainer.explain()
                print(d, m, e, 'auc: ', auc, 'auc_std: ', auc_std, 'rdd: ', rdd, 'rdd_std: ',
                      rdd_std)

    explainers = ['NAMPGE']
    loss_type = 'ib'
    for d in datasets:
        for e in explainers:
            for m in models:
                explainer = MainExplainer(d, m, e, wandb_log=False, loss_type=loss_type, force_run=True,
                                          save_explaining_results=True)
                auc, auc_std, rdd, rdd_std = explainer.explain()
                print(d, m, e, 'auc: ', auc, 'auc_std: ', auc_std, 'rdd: ', rdd, 'rdd_std: ',
                      rdd_std)


def do_log_analyzing():
    explainers = ['GNNE']
    models = ['GraphGCN']
    datasets = ['ba2motif']
    loss_type = 'ib'
    for e in explainers:
        for m in models:
            for d in datasets:
                explainer = MainExplainer(d, m, e, wandb_log=True, loss_type=loss_type)
                explainer.force_run = True
                auc, auc_std, rdd, rdd_std = explainer.explain()
                print(d, m, e, 'auc: ', auc, 'auc_std: ', auc_std, 'rdd: ', rdd, 'rdd_std: ',
                      rdd_std)


def run_grad():
    datasets = ['bareg1', 'bareg2', 'triangles', 'crippen']
    explainers = ['GRAD']
    models = ['GraphGCN']
    for d in datasets:
        for e in explainers:
            for m in models:
                explainer = MainExplainer(d, m, e, wandb_log=False, force_run=True,
                                          save_explaining_results=True)
                auc, auc_std, rdd, rdd_std = explainer.explain()
                print(d, m, e, 'auc: ', auc, 'auc_std: ', auc_std, 'rdd: ', rdd, 'rdd_std: ',
                      rdd_std)


def run_gat():
    datasets = ['bareg1', 'bareg2', 'triangles', 'crippen']
    explainers = ['GAT']
    models = ['GAT']
    for d in datasets:
        for e in explainers:
            for m in models:
                explainer = MainExplainer(d, m, e, wandb_log=False, force_run=True,
                                          save_explaining_results=True)
                auc, auc_std, rdd, rdd_std = explainer.explain()
                print(d, m, e, 'auc: ', auc, 'auc_std: ', auc_std, 'rdd: ', rdd, 'rdd_std: ',
                      rdd_std)


def run_ctl():

    datasets = ['bareg2', 'triangles', 'bareg1']
    explainers = ['CTLALTMIXPGE', 'CTLPGE', 'CTLMIXSFTPGE']
    models = ['GraphGCN']
    force_run = True
    loss_type = 'ctl'
    for d in datasets:
        for e in explainers:
            for m in models:
                explainer = MainExplainer(d, m, e, wandb_log=False, loss_type=loss_type, force_run=force_run,
                                          save_explaining_results=True)
                auc, auc_std, rdd, rdd_std = explainer.explain()
                print(d, m, e, 'auc: ', auc, 'auc_std: ', auc_std, 'rdd: ', rdd, 'rdd_std: ',
                      rdd_std)

    datasets = ['crippen']
    explainers = ['CTLPGE', 'CTLMIXSFTPGE', 'CTLALTMIXPGE']
    models = ['GraphGCN']
    force_run = True
    loss_type = 'ctl'
    for d in datasets:
        for e in explainers:
            for m in models:
                explainer = MainExplainer(d, m, e, wandb_log=False, loss_type=loss_type, force_run=force_run,
                                          save_explaining_results=True)
                auc, auc_std, rdd, rdd_std = explainer.explain()
                print(d, m, e, 'auc: ', auc, 'auc_std: ', auc_std, 'rdd: ', rdd, 'rdd_std: ',
                      rdd_std)


if __name__ == '__main__':
    run_ctl()

