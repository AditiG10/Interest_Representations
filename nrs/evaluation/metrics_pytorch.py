import torch
import torch.nn.functional as F
import torchmetrics

# Helper function to handle both single and multi-interest dimensions
def handle_multi_interest_dimensions(y_true, y_score):
    if y_true.dim() > 1:
        y_score = torch.max(y_score, dim=-1).values
        y_true = torch.max(y_true, dim=-1).values
    return y_true, y_score

# Ranking Metrics

def dcg_score(y_true, y_score, k=10):
    y_true, y_score = handle_multi_interest_dimensions(y_true, y_score)
    order = torch.argsort(y_score, descending=True)
    y_true = torch.gather(y_true, 0, order[:k])
    gains = 2 ** y_true - 1
    discounts = torch.log2(torch.arange(1, k+1, device=y_true.device).float() + 1)
    return torch.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    y_true, y_score = handle_multi_interest_dimensions(y_true, y_score)
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best if best != 0 else torch.tensor(0.0, device=y_true.device)

def rr_score(y_true, y_score):
    y_true, y_score = handle_multi_interest_dimensions(y_true, y_score)
    order = torch.argsort(y_score, descending=True)
    y_true_sorted = torch.gather(y_true, 0, order)
    rr_score = y_true_sorted / (torch.arange(len(y_true_sorted), device=y_true_sorted.device, dtype=torch.float) + 1)
    return torch.max(rr_score).item() if torch.sum(y_true_sorted) > 0 else 0.0

def ctr_score(y_true, y_score, k=1):
    y_true, y_score = handle_multi_interest_dimensions(y_true, y_score)
    order = torch.argsort(y_score, descending=True)
    y_true = torch.gather(y_true, 0, order[:k])
    return torch.mean(y_true.float())

# Classification Metrics

def acc_score(y_true, y_score):
    y_true, y_score = handle_multi_interest_dimensions(y_true, y_score)
    y_pred = torch.round(torch.clamp(y_score, 0, 1))
    return torchmetrics.functional.accuracy(y_pred, y_true)

def recall_score(y_true, y_score):
    y_true, y_score = handle_multi_interest_dimensions(y_true, y_score)
    y_pred = torch.round(torch.clamp(y_score, 0, 1))
    return torchmetrics.functional.recall(y_pred, y_true)

def precision_score(y_true, y_score):
    y_true, y_score = handle_multi_interest_dimensions(y_true, y_score)
    y_pred = torch.round(torch.clamp(y_score, 0, 1))
    return torchmetrics.functional.precision(y_pred, y_true)

def confusion_matrix(y_true, y_score):
    y_true, y_score = handle_multi_interest_dimensions(y_true, y_score)
    y_pred = torch.round(torch.clamp(y_score, 0, 1))
    return torchmetrics.functional.confusion_matrix(y_pred, y_true)

# Regression Metrics (unchanged as they generally handle 1D predictions directly)

def mse_score(y_true, y_pred):
    return F.mse_loss(y_pred, y_true)

def mae_score(y_true, y_pred):
    return F.l1_loss(y_pred, y_true)

def r2_score(y_true, y_pred):
    return torchmetrics.functional.r2_score(y_pred, y_true)

def corr_score(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return torchmetrics.functional.pearson_correlation(y_pred, y_true)
