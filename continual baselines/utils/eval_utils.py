import torch
import numpy
from sklearn.metrics import classification_report


class EvalUtils:
    def __init__(self):
        pass

    @staticmethod
    def evaluate_model(model, device, loss_fn, loader):
        pass


class NaiveEval(EvalUtils):
    def __init__(self):
        super().__init__()

    @staticmethod
    def evaluate_model(model, device, loss_fn, loader, gpu=None):
        if loader.dataset.train:
            print("\n[EVAL ON TRAINING DATA]")
        else:
            print("[EVAL ON TESTING DATA]")

        num_correct = 0
        num_samples = 0
        model.eval()

        with torch.no_grad():
            for _, (image, label) in enumerate(loader):
                if device is not None:
                    image, label = image.to(device), label.to(device)
                else:
                    if gpu is not None:
                        image = image.cuda(gpu, non_blocking=True)
                    if torch.cuda.is_available():
                        label = label.cuda(gpu, non_blocking=True)
                scores = model(image)

                try:
                    _, pred = scores.max(dim=1)
                except AttributeError:
                    _, pred = scores[0].max(dim=1)
                num_correct += (pred == label).sum()
                num_samples += pred.size(0)

            accuracy = num_correct / num_samples

        return accuracy


class LSTMEval(EvalUtils):
    def __init__(self):
        super().__init__()

    @staticmethod
    def evaluate_model(model, device, loss_fn, loader):
        print("[EVAL ON TEST DATA]")

        num_correct = 0
        num_samples = 0
        model.eval()

        with torch.no_grad():
            for _, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                scores = model(x)

                try:
                    _, pred = scores.max(dim=1)
                except AttributeError:
                    _, pred = scores[0].max(dim=1)
                num_correct += (pred == y).sum()
                num_samples += pred.size(0)

            accuracy = num_correct / num_samples

        return accuracy


class MTLEval(EvalUtils):
    def __init__(self):
        super().__init__()

    @staticmethod
    def evaluate_model(model, device, loss_fn, loader):
        print("\n[EVALUATION]")

        num_correct_obj, num_samples_obj = 0, 0
        num_correct_sty, num_samples_sty = 0, 0
        y_pred_obj, y_pred_sty = torch.zeros(0, dtype=torch.long, device='cpu'), torch.zeros(0, dtype=torch.long,
                                                                                             device='cpu')
        y_true_obj, y_true_sty = torch.zeros(0, dtype=torch.long, device='cpu'), torch.zeros(0, dtype=torch.long,
                                                                                             device='cpu')
        model.eval()

        with torch.no_grad():
            for _, (image, label_obj, label_sty) in enumerate(loader):
                image, label_obj, label_sty = image.to(device), label_obj.to(device), label_sty.to(device)
                scores_obj, scores_sty = model(image)

                (_, pred_obj), (_, pred_sty) = scores_obj.max(dim=1), scores_sty.max(dim=1)
                num_correct_obj += (pred_obj == label_obj).sum()
                num_correct_sty += (pred_sty == label_sty).sum()
                num_samples_obj += pred_obj.size(0)
                num_samples_sty += pred_sty.size(0)

                y_pred_obj, y_pred_sty = (
                    torch.cat([y_pred_obj, pred_obj.view(-1).cpu()]),
                    torch.cat([y_pred_sty, pred_sty.view(-1).cpu()])
                )
                y_true_obj, y_true_sty = (
                    torch.cat([y_true_obj, label_obj.view(-1).cpu()]),
                    torch.cat([y_true_sty, label_sty.view(-1).cpu()])
                )

            accuracy_obj, accuracy_sty = num_correct_obj / num_samples_obj, num_correct_sty / num_samples_sty
            cr_obj, cr_sty = (
                classification_report(y_true_obj.numpy(), y_pred_obj.numpy(), labels=[i for i in range(10)]),
                classification_report(y_true_sty.numpy(), y_pred_sty.numpy(), labels=[i for i in range(10)])
            )

            return accuracy_obj, accuracy_sty, cr_obj, cr_sty


class EvalF:
    '''
    helper utility for calculating average F-score
    '''
    def __init__(self, t1_accuracy_dict, avg_accuracy_dict):
        self.t1_accuracy_dict = t1_accuracy_dict
        self.avg_accuracy_dict = avg_accuracy_dict
        self.len = len(list(self.avg_accuracy_dict.values()))

    def __call__(self):
        fscore = 0.0
        for val_avg, val_t1 in zip(self.avg_accuracy_dict.values(), self.t1_accuracy_dict.values()):
            fscore += (2 * val_avg[-1]) / ((val_t1[0] - val_t1[-1]) * val_avg[-1] + 1)
        
        print("average F: {}".format(fscore/self.len))
