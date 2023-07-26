from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


class EWC_MTL:
    def __init__(self):
        pass

    @staticmethod
    def on_task_update(
            optpar_dict, fisher_dict, task_id, dataloader, model, loss_fn, optimizer, device
    ):

        model.train()
        optimizer.zero_grad()

        # accumulating gradients
        for image, label_obj, label_sty in dataloader:
            image, label_obj, label_sty = image.to(device), label_obj.to(device), label_sty.to(device)
            output_obj, output_sty = model(image)
            loss_obj, loss_sty = loss_fn(output_obj, label_obj), loss_fn(output_sty, label_sty)
            loss = loss_obj + loss_sty
            loss.backward()

        fisher_dict[task_id] = {}
        optpar_dict[task_id] = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                optpar_dict[task_id][name] = param.data.clone()
                fisher_dict[task_id][name] = param.grad.data.clone().pow(2)

        return optpar_dict, fisher_dict




