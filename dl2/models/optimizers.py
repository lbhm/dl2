# License not provided by author
# Taken from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/image_classification/optimizers.py

import torch


def get_optimizer(parameters, lr, args, state=None):
    if args.optimizer == 'sgd':
        optimizer = get_sgd_optimizer(
            parameters, lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov,
            bn_weight_decay=args.bn_weight_decay
        )
    elif args.optimizer == 'rmsprop':
        optimizer = get_rmsprop_optimizer(
            parameters, lr, alpha=args.rmsprop_alpha, momentum=args.momentum, weight_decay=args.weight_decay,
            eps=args.rmsprop_eps, bn_weight_decay=args.bn_weight_decay
        )
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer} is not implemented!")

    if state is not None:
        optimizer.load_state_dict(state)

    return optimizer


def get_sgd_optimizer(parameters, lr, momentum, weight_decay, nesterov=False, bn_weight_decay=False, verbose=False):
    if bn_weight_decay:
        params = [v for n, v in parameters]
        if verbose:
            print("[INFO] Weight decay applied to BN parameters.")
    else:
        bn_params = [v for n, v in parameters if "bn" in n]
        rest_params = [v for n, v in parameters if "bn" not in n]
        params = [
            {"params": bn_params, "weight_decay": 0},
            {"params": rest_params, "weight_decay": weight_decay},
        ]
        if verbose:
            print("[INFO] Weight decay NOT applied to BN parameters.")

    optimizer = torch.optim.SGD(params, lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)

    return optimizer


def get_rmsprop_optimizer(parameters, lr, alpha, weight_decay, momentum, eps, bn_weight_decay=False):
    bn_params = [v for n, v in parameters if "bn" in n]
    rest_params = [v for n, v in parameters if not "bn" in n]

    params = [
        {"params": bn_params,  "weight_decay": weight_decay if bn_weight_decay else 0},
        {"params": rest_params, "weight_decay": weight_decay},
    ]

    optimizer = torch.optim.RMSprop(params, lr=lr, alpha=alpha, weight_decay=weight_decay, momentum=momentum, eps=eps)

    return optimizer
