import torch

def dice_no_threshold(
    outputs:torch.Tensor,
    targets:torch.Tensor,
    eps:float = 1e-6,
    threshold:float=0.5,
):
    """
    dice score
    outputs: prediction
    targets: ground truth
    Reference:
    https://catalyst-team.github.io/catalyst/_modules/catalyst/dl/utils/criterion/dice.html
    """
    if threshold is not None:
        outputs = (outputs > threshold).float()

    intersection = torch.sum(targets * outputs)
    union = torch.sum(targets) + torch.sum(outputs)
    return 2 * intersection / (union + eps)


def get_accuracy(outputs:torch.Tensor,
                 targets:torch.Tensor,
                 threshold:float=0.5):
    """Accuracy for gray scale segmentation

    Args:
        outputs (torch.Tensor): prediction
        targets (torch.Tensor): ground truth
        threshold (float, optional): Threshold for classification. Defaults to None.
    """
    if threshold is not None:
        outputs = (outputs > threshold).float()

    correct = torch.sum(outputs==targets)
    tensor_size = outputs.size(0)*outputs.size(1)*outputs.size(2)*outputs.size(3)
    return float(correct)/float(tensor_size)

def get_recall(outputs:torch.Tensor,
               targets:torch.Tensor,
               threshold :float=0.5):
    if threshold is not None:
        outputs = (outputs > threshold).float()
    
    
    # TP: True positive
    # FN: False Negative
    TP = ((outputs==1).float()+(targets==1).float())==2
    FN = ((outputs==0).float()+(targets==1).float())==2
    
    #return float(torch.sum(TP))
    return float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

def get_specificity(outputs:torch.Tensor,
                    targets:torch.Tensor,
                    threshold:float=0.5):
    if threshold is not None:
        outputs = (outputs > threshold).float()
    
    
    # TN : True negative
    # FP : False positive
    TN = ((outputs==0).float()+(targets==0).float())==2
    FP = ((outputs==1).float()+(targets==0).float())==2
    
    #return float(torch.sum(TN))
    return float(torch.sum(TN)) / (float(torch.sum(TN+FP)) + 1e-6)

def get_precision(outputs:torch.Tensor,
                  targets:torch.Tensor,
                  threshold:float=0.5):
    if threshold is not None:
        outputs = (outputs > threshold).float()
    TP = ((outputs==1).float()+(targets==1).float())==2
    FP = ((outputs==1).float()+(targets==0).float())==2
    
    return float(torch.sum(TP)) / (float(torch.sum(TP+FP)) + 1e-6)
    
def get_F1(outputs:torch.Tensor,
           targets:torch.Tensor,
           threshold:float=0.5):
    sensitivity = get_recall(outputs,targets,threshold=threshold)
    precision = get_precision(outputs,targets,threshold=threshold)
    
    return 2*sensitivity*precision/(sensitivity+precision+1e-6)
    
def get_JS(outputs:torch.Tensor,
           targets:torch.Tensor,
           threshold:float=0.5):
    #JS: Jaccard similarity
    if threshold is not None:
        outputs = (outputs > threshold).float()
        
    inter = torch.sum((outputs+targets)==2)
    union = torch.sum((outputs+targets)>=1)
    
    return float(inter) / (float(union) + 1e-6)


    
    

        
    
# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'dice_no_threshold': dice_no_threshold,
    'accuracy': get_accuracy,
    'recall': get_recall,
    'precision': get_precision,
    'specificity': get_specificity,
    'F1 score': get_F1,
    'Jaccard similarity': get_JS
    # could add more metrics such as accuracy for each token type
}