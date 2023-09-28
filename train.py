# Load the libraries.
import os 
import numpy as np
import torch
from tqdm import tqdm
import logging
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

# Load local files
import model.u_net as u_net
from util.data_loader import ActiveWFDataset
import util.loss_func as loss_func
from util.optimzer_RAdam import RAdam
import util.toolbox
from util.evaluate import evaluate
import util.metric as metric



# Train
def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """train the model

    Args:
        model (torch.nn.Module): the neural network
        optimizer (torch.optim): optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader (DataLoader): a torch.utils.data.DataLoader object that fetches training data
        metrics (dict): a dictionary of functions that compute a metric using the output and labels of each batch
        params (params.json): hyperparameters
    """
    
    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = util.toolbox.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if availabel
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(
                    non_blocking=True), labels_batch.cuda(non_blocking=True)

            # compute model output and loss
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            #clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu()
                labels_batch = labels_batch.data.cpu()

                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

        # compute mean of all metrics in summary
        metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
        logging.info(f"- Train metrics: {metrics_string}")
    return metrics_mean

def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer,
                       loss_fn, metrics, params, model_dir, restore_file=None,
                       tensorboard_dir='./model/test'):
    """Train the model and evaluate every epoch

    Args:
        model (torch.nn.Module): the neural network
        train_dataloader (DataLoader): a torch.utils.data.DataLoader object that fetches training data
        val_dataloader (DataLoader): a torch.utils.data.DataLoader object that fetches validation data
        optimizer (torch.optim): optimizer for parameters of model
        loss_fn : a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics (dict): a dictionary of functions that compute a metric using the output and labels of each batch
        params (Params): hyperparameters
        model_dir (string): directory containing config, weights and log
        restore_file (string, optional): optional-name of file to restore from (without its extention .pth.tar).
    """
    
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(model_dir, f'{restore_file}.pth.tar')
        logging.info(f"Restoring parameters from {restore_path}")
        util.toolbox.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=2, cooldown=2)

    # for test
    # TODO: These outputs are for testing.
    train_loss_list = []
    valid_loss_list = []
    dice_score_list = []
    lr_rate_list = []

    writer = SummaryWriter(log_dir=tensorboard_dir, flush_secs=120)
    
    for epoch in range(params.num_epochs):
        
        train_loss = 0.0
        valid_loss = 0.0
        dice_score = 0.0
        
        # Run one epoch
        logging.info(f"Epoch {epoch + 1}/{params.num_epochs}")

        # One full pass over the training set
        train_metrics = train(model, optimizer, loss_fn, train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)

        writer.add_scalars(main_tag='Dice Score',
                          tag_scalar_dict={'train':train_metrics['dice_no_threshold'],
                                           'validation':val_metrics['dice_no_threshold']},
                          global_step=epoch)
        writer.add_scalars(main_tag='Accuracy',
                          tag_scalar_dict={'train':train_metrics['accuracy'],
                                           'validation':val_metrics['accuracy']},
                          global_step=epoch)
        writer.add_scalars(main_tag='Recall',
                          tag_scalar_dict={'train':train_metrics['recall'],
                                           'validation':val_metrics['recall']},
                          global_step=epoch)
        writer.add_scalars(main_tag='Precision',
                          tag_scalar_dict={'train':train_metrics['precision'],
                                           'validation':val_metrics['precision']},
                          global_step=epoch)
        writer.add_scalars(main_tag='Specificity',
                          tag_scalar_dict={'train':train_metrics['specificity'],
                                           'validation':val_metrics['specificity']},
                          global_step=epoch)
        writer.add_scalars(main_tag='F1 score',
                          tag_scalar_dict={'train':train_metrics['F1 score'],
                                           'validation':val_metrics['F1 score']},
                          global_step=epoch)
        writer.add_scalars(main_tag='Jaccard similarity',
                          tag_scalar_dict={'train':train_metrics['Jaccard similarity'],
                                           'validation':val_metrics['Jaccard similarity']},
                          global_step=epoch)
        
        val_acc = val_metrics['dice_no_threshold']
        is_best = val_acc >= best_val_acc

        # Save weights
        util.toolbox.save_checkpoint({'epoch': epoch + 1,
                                      'state_dict': model.state_dict(),
                                      'optim_dict': optimizer.state_dict()},
                                     is_best=is_best,
                                     checkpoint=model_dir)

        # TODO: calculate average losses
        train_loss = train_metrics['loss']
        valid_loss = val_metrics['loss']
        dice_score = val_acc
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        dice_score_list.append(dice_score)
        lr_rate_list.append([param_group['lr'] for param_group in optimizer.param_groups])


        # If it is the best, save the model parameters
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            util.toolbox.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        util.toolbox.save_dict_to_json(val_metrics, last_json_path)

        scheduler.step(val_metrics['loss'])
        
    writer.close()
    
    # TODO: Save these stuff.
    Learning_plot = {
    "lr_rate_list": lr_rate_list, 
    "train_loss_list": train_loss_list,
    "valid_loss_list": valid_loss_list, 
    "dice_score_list": dice_score_list
    }
    learning_plot_data = pd.DataFrame(Learning_plot)
    training_stats = os.path.join(
            model_dir, "training_statistics_test.csv")
    learning_plot_data.to_csv(training_stats)
    
        
if __name__ == '__main__':
    # training/validation data folder path
    train_features_dir = 'C:/Users/yang.xin1022/Desktop/train_test2/train/raw'
    train_labels_dir = 'C:/Users/yang.xin1022/Desktop/train_test2/train/label'
    valid_features_dir = 'C:/Users/yang.xin1022/Desktop/train_test2/valid/raw'
    valid_labels_dir = 'C:/Users/yang.xin1022/Desktop/train_test2/valid/label'
    # Load the parameters from json file
    model_dir = 'C:/Users/yang.xin1022/Desktop/train_test2/raw_test'
    param_dir = 'E:/my-whitecaps/code/organized/model'
    # Log dir
    log_dir = 'E:/my-whitecaps/code/organized/log'
    log_name = 'train_raw_totoal_W.log'
    tensorboard_dir = './model/test/raw_test_totoal_W'
    # "Directory containing params.json"
    json_path = os.path.join(param_dir, 'params.json')
    restore_file = None
    
    # Optional, name of the file in --model_dir containing weights to reload before training
    assert os.path.isfile(
        json_path
    ), f"No json configuration file found at {json_path}"
    params = util.toolbox.Params(json_path)


    # check whether GPU is available
    params.cuda = torch.cuda.is_available()
    print(f'GPU is available: {params.cuda}')
    print(f'The number of images in each iteration: {params.batch_size}')


    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    util.toolbox.set_logger(os.path.join(log_dir, log_name))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # Build dataloader
    active_train = ActiveWFDataset(train_features_dir, train_labels_dir)
    train_dl = torch.utils.data.DataLoader(active_train, params.batch_size,
                                         shuffle=True,drop_last=True,
                                         num_workers=params.num_workers,
                                         pin_memory=params.cuda)
    active_valid = ActiveWFDataset(valid_features_dir, valid_labels_dir)
    val_dl = torch.utils.data.DataLoader(active_valid, params.batch_size,
                                         shuffle=False,drop_last=True,
                                         num_workers=params.num_workers,
                                         pin_memory=params.cuda)

    logging.info("- done.")

    # Define the model and optimizer
    model = u_net.UNet(params).float().cuda() if params.cuda else u_net.UNet(params).float()
    optimizer = RAdam(model.parameters(), lr=params.learning_rate)


    # fetch loss function and metrics
    loss_fn = loss_func.BCEDiceLoss(eps=1.0, activation=None)
    metrics = metric.metrics

    # Train the model
    logging.info(f"Starting training for {params.num_epochs} epoch(s)")
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn,
                       metrics, params, model_dir, restore_file, tensorboard_dir)
    