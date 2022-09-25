import os
import torch
from torch import optim
import torch.nn.functional as F
import deepnovo_config
from data_reader import DeepNovoTrainDataset, collate_func
from model import DeepNovoModel, device, InitNet
import time
import math
import logging
import numpy as np

#default.device = torch.device('cpu')
logger = logging.getLogger(__name__)

forward_model_save_name = 'forward_deepnovo.pth'
backward_model_save_name = 'backward_deepnovo.pth'
init_net_save_name = 'init_net.pth'

logger.info(f"using device: {device}")
#print("devive = ",device)
device2 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def to_one_hot(y, n_dims=None):
    """ Take integer y with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot


def focal_loss(logits, labels, ignore_index=-100, gamma=2.):
    #print("labels",labels.shape)
    #print("logits",logits.shape)
    """

    :param logits: float tensor of shape [batch, T, 26]
    :param labels: long tensor of shape [batch, T]
    :param ignore_index: ignore the loss of those tokens
    :param gamma:
    :return: average loss, num_valid_token
    """
    valid_token_mask = (labels != ignore_index).float()  # [batch, T]
    #print(valid_token_mask.shape)
    num_valid_token = torch.sum(valid_token_mask)
    #print("num valis tokens",num_valid_token)
    batch_size, T, num_classes = logits.size()
    sigmoid_p = torch.sigmoid(logits)
    #print(sigmoid_p.shape)
    target_tensor = to_one_hot(labels, n_dims=num_classes).float().to(device)
    #print(target_tensor.shape)
    zeros = torch.zeros_like(sigmoid_p)
    #print(zeros.shape)
    pos_p_sub = torch.where(target_tensor >= sigmoid_p, target_tensor - sigmoid_p, zeros)  # [batch, T, 26]
    #print(pos_p_sub.shape)
    neg_p_sub = torch.where(target_tensor > zeros, zeros, sigmoid_p)  # [batch, T, 26]
    #print(neg_p_sub.shape)
    per_token_loss = - (pos_p_sub ** gamma) * torch.log(torch.clamp(sigmoid_p, 1e-8, 1.0)) - \
                     (neg_p_sub ** gamma) * torch.log(torch.clamp(1.0 - sigmoid_p, 1e-8, 1.0))
    #print(per_token_loss.shape)
    per_entry_loss = torch.sum(per_token_loss, dim=2)  # [batch, T]
    #print(per_entry_loss.shape)
    per_entry_loss = per_entry_loss * valid_token_mask  # masking out loss from pad tokens
    #print(per_entry_loss.shape)
    per_entry_average_loss = torch.sum(per_entry_loss) / (num_valid_token + 1e-6)
    #print(per_entry_average_loss.shape)
    return per_entry_average_loss, num_valid_token


def build_model(training=True):
    """

    :return:
    """
    forward_deepnovo = DeepNovoModel()
    backward_deepnovo = DeepNovoModel()
    if deepnovo_config.use_lstm:
        init_net = InitNet()
    else:
        init_net = None

    # load pretrained params if exist
    if os.path.exists(os.path.join(deepnovo_config.train_dir, forward_model_save_name)):
        assert os.path.exists(os.path.join(deepnovo_config.train_dir, backward_model_save_name))
        logger.info("load pretrained model")
        forward_deepnovo.load_state_dict(torch.load(os.path.join(deepnovo_config.train_dir, forward_model_save_name),
                                                    map_location=device))
        backward_deepnovo.load_state_dict(torch.load(os.path.join(deepnovo_config.train_dir, backward_model_save_name),
                                                     map_location=device))
        if deepnovo_config.use_lstm:
            init_net.load_state_dict(torch.load(os.path.join(deepnovo_config.train_dir, init_net_save_name),
                                                map_location=device))
    else:
        assert training, f"building model for testing, but could not found weight under directory " \
                         f"{deepnovo_config.train_dir}"
        logger.info("initialize a set of new parameters")

    if deepnovo_config.use_lstm:
        # share embedding matrix
        backward_deepnovo.embedding.weight = forward_deepnovo.embedding.weight

    backward_deepnovo = backward_deepnovo.to(device)
    forward_deepnovo = forward_deepnovo.to(device)
    if deepnovo_config.use_lstm:
        init_net = init_net.to(device)
    return forward_deepnovo, backward_deepnovo, init_net


def extract_and_move_data(data):
    """

    :param data: result from dataloader
    :return:
    """
    peak_location, \
    peak_intensity, \
    spectrum_representation,\
    batch_forward_id_target, \
    batch_backward_id_target, \
    batch_forward_ion_index, \
    batch_backward_ion_index, \
    batch_forward_id_input, \
    batch_backward_id_input = data

    # move to device
    peak_location = peak_location.to(device)
    peak_intensity = peak_intensity.to(device)
    spectrum_representation = spectrum_representation.to(device)
    batch_forward_id_target = batch_forward_id_target.to(device)
    batch_backward_id_target = batch_backward_id_target.to(device)
    batch_forward_ion_index = batch_forward_ion_index.to(device)
    batch_backward_ion_index = batch_backward_ion_index.to(device)
    batch_forward_id_input = batch_forward_id_input.to(device)
    batch_backward_id_input = batch_backward_id_input.to(device)
    return (peak_location,
            peak_intensity,
            spectrum_representation,
            batch_forward_id_target,
            batch_backward_id_target,
            batch_forward_ion_index,
            batch_backward_ion_index,
            batch_forward_id_input,
            batch_backward_id_input
            )


def validation(forward_deepnovo, backward_deepnovo, init_net, valid_loader) -> float:
    with torch.no_grad():
        valid_loss = 0
        num_valid_samples = 0
        max_num = 2000
        for data in valid_loader:
            #if np.random.random()>0.002: # approx 1000 spectra
            #   continue
            peak_location, \
            peak_intensity, \
            spectrum_representation, \
            batch_forward_id_target, \
            batch_backward_id_target, \
            batch_forward_ion_index, \
            batch_backward_ion_index, \
            batch_forward_id_input, \
            batch_backward_id_input = extract_and_move_data(data)
            batch_size = batch_backward_id_target.size(0)
            if deepnovo_config.use_lstm:
                initial_state_tuple = init_net(spectrum_representation)
                forward_logit, _ = forward_deepnovo(batch_forward_ion_index, peak_location, peak_intensity,
                                                    batch_forward_id_input, initial_state_tuple)
                backward_logit, _ = backward_deepnovo(batch_backward_ion_index, peak_location, peak_intensity,
                                                      batch_backward_id_input, initial_state_tuple)
            else:
                forward_logit = forward_deepnovo(batch_forward_ion_index, peak_location, peak_intensity)
                backward_logit = backward_deepnovo(batch_backward_ion_index, peak_location, peak_intensity)
            forward_loss, f_num = focal_loss(forward_logit, batch_forward_id_target, ignore_index=0, gamma=2.)
            backward_loss, b_num = focal_loss(backward_logit, batch_backward_id_target, ignore_index=0, gamma=2.)
            valid_loss += forward_loss.item() * f_num.item() + backward_loss.item() * b_num.item()
            num_valid_samples += f_num.item() + b_num.item()
            if num_valid_samples>max_num:
            	break
    print(num_valid_samples)
    average_valid_loss = valid_loss / (num_valid_samples + 1e-6)
    return float(average_valid_loss)


def perplexity(log_loss):
    return math.exp(log_loss) if log_loss < 300 else float('inf')


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 3 epochs"""
    lr = deepnovo_config.init_lr * (0.1 ** ((epoch + 1) // 3))
    logger.info(f"epoch: {epoch}\tlr: {lr}")
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_model(forward_deepnovo, backward_deepnovo, init_net):
    torch.save(forward_deepnovo.state_dict(), os.path.join(deepnovo_config.train_dir,
                                                           forward_model_save_name))
    torch.save(backward_deepnovo.state_dict(), os.path.join(deepnovo_config.train_dir,
                                                            backward_model_save_name))
    if deepnovo_config.use_lstm:
        torch.save(init_net.state_dict(), os.path.join(deepnovo_config.train_dir,
                                                   init_net_save_name))


def train():

    
    
    log_file = os.getcwd()+"/my_log2.txt"
    log_file_handle = open(log_file, 'w+')
    #print("test",file=log_file_handle)
    #print("batch size: ",deepnovo_config.batch_size,file=log_file_handle)
    
    
    train_set = DeepNovoTrainDataset(deepnovo_config.input_feature_file_train,
                                     deepnovo_config.input_spectrum_file_train)
    num_train_features = len(train_set)
    steps_per_epoch = int(num_train_features / deepnovo_config.batch_size)
    #print("steps per epoch = ",steps_per_epoch,file=log_file_handle)
    logger.info(f"{steps_per_epoch} steps per epoch")
    train_data_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                    batch_size=deepnovo_config.batch_size,
                                                    shuffle=True,
                                                    num_workers=deepnovo_config.num_workers,
                                                    collate_fn=collate_func)
    valid_set = DeepNovoTrainDataset(deepnovo_config.input_feature_file_valid,
                                     deepnovo_config.input_spectrum_file_valid)
    valid_data_loader = torch.utils.data.DataLoader(dataset=valid_set,
                                                    batch_size=deepnovo_config.batch_size,
                                                    shuffle=False,
                                                    num_workers=deepnovo_config.num_workers,
                                                    collate_fn=collate_func)
    forward_deepnovo, backward_deepnovo, init_net = build_model()
    # sparse_params = forward_deepnovo.spectrum_embedding_matrix.parameters()
    dense_params = list(forward_deepnovo.parameters()) + list(backward_deepnovo.parameters())

    dense_optimizer = optim.Adam(dense_params,
                                 lr=deepnovo_config.init_lr,
                                 weight_decay=deepnovo_config.weight_decay)
    # sparse_optimizer = optim.SparseAdam(sparse_params, lr=deepnovo_config.init_lr)
    dense_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(dense_optimizer, 'min', factor=0.5, verbose=True,
                                                                 threshold=1e-4, cooldown=10, min_lr=1e-5)
    # sparse_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(sparse_optimizer, 'min', factor=0.25, verbose=False,
    #                                                        threshold=1e-4, cooldown=10, min_lr=1e-5)
    
#    model_parameters = filter(lambda p: p.requires_grad, forward_deepnovo.parameters())
#    params = sum([np.prod(p.size()) for p in model_parameters])
    #print(params)
#    model_parameters = filter(lambda p: p.requires_grad, backward_deepnovo.parameters())
#    params = sum([np.prod(p.size()) for p in model_parameters])
    #print(params)
#    model_parameters = filter(lambda p: p.requires_grad, init_net.parameters())
#    params = sum([np.prod(p.size()) for p in model_parameters])
    #print(params)
    
    best_valid_loss = float("inf")
    # train loop
    best_epoch = None
    best_step = None
    start_time = time.time()

    avg_train_loss = []
    avg_test_loss = []
    for epoch in range(deepnovo_config.num_epoch):
        # learning rate schedule
        # adjust_learning_rate(optimizer, epoch)
        #print("current epoch: {0}".format(epoch),file=log_file_handle)
        train_losses =[]
        test_losses = []
        
        # 
        
        for i, data in enumerate(train_data_loader):
            dense_optimizer.zero_grad()
            # sparse_optimizer.zero_grad()


            peak_location, \
            peak_intensity, \
            spectrum_representation, \
            batch_forward_id_target, \
            batch_backward_id_target, \
            batch_forward_ion_index, \
            batch_backward_ion_index, \
            batch_forward_id_input, \
            batch_backward_id_input = extract_and_move_data(data)
            batch_size = batch_backward_id_target.size(0)
#            print((peak_location.type()))
#            print((peak_intensity.type()))
#            print((spectrum_representation.type()))
#            print((batch_forward_id_target.type()))
#            print((batch_forward_ion_index.type()))
#            print((batch_forward_id_input.type()))
            #shape=peak_location.shape 
            #peak_location = torch.zeros(shape[0], shape[1]).float()
            shape=batch_forward_ion_index.shape 
            #batch_forward_ion_index = torch.zeros(shape).float()
            #print("peak locatiomn£",peak_location.shape)
            if deepnovo_config.use_lstm:
                initial_state_tuple = init_net(spectrum_representation)
                #print(initial_state_tuple[0].size(),initial_state_tuple[1].size())
                forward_logit, nst = forward_deepnovo(batch_forward_ion_index, peak_location, peak_intensity,
                                                    batch_forward_id_input, initial_state_tuple)

                #print(forward_logit.size())
                #print(nst[0].size(),nst[1].size())		

                backward_logit, _ = backward_deepnovo(batch_backward_ion_index, peak_location, peak_intensity,
                                                      batch_backward_id_input, initial_state_tuple)
            else:
                forward_logit = forward_deepnovo(batch_forward_ion_index, peak_location, peak_intensity)
                backward_logit = backward_deepnovo(batch_backward_ion_index, peak_location, peak_intensity)

            forward_loss, _ = focal_loss(forward_logit, batch_forward_id_target, ignore_index=0, gamma=2.)
            backward_loss, _ = focal_loss(backward_logit, batch_backward_id_target, ignore_index=0, gamma=2.)
            total_loss = (forward_loss + backward_loss) / 2.
            
            ##print(forward_logit.size())
            #print(i)
            #print(total_loss)
            ##print("batch num: {0}".format(i),total_loss,file=log_file_handle)
            train_losses.append(total_loss.detach().cpu().numpy())
            ## compute gradient
            total_loss.backward()

            #print(batch_forward_id_target.detach().cpu().numpy()[0])
            #print([np.argmax(x) for x in forward_logit.detach().cpu().numpy()[0]])



            # clip gradient
            # torch.nn.utils.clip_grad_norm_(dense_params, deepnovo_config.max_gradient_norm)
	    if (i+1) % deepnovo_config.accumulation_steps == 0:             # Wait for several backward steps
            	dense_optimizer.step()                            # Now we can do an optimizer step
            	dense_optimizer.zero_grad()
        
        #    dense_optimizer.step()
            # sparse_optimizer.step()

            if (i + 1) % deepnovo_config.steps_per_validation == 0:
                duration = time.time() - start_time
                step_time = duration / deepnovo_config.steps_per_validation
                loss_cpu = total_loss.item()
                # evaluation mode
                eval_start_time = time.time()
                forward_deepnovo.eval()
                backward_deepnovo.eval()
                validation_loss = validation(forward_deepnovo, backward_deepnovo, init_net, valid_data_loader)
                dense_scheduler.step(validation_loss)
                # sparse_scheduler.step(validation_loss)
                test_losses.append(validation_loss)
                logger.info(f"epoch {epoch} step {i}/{steps_per_epoch}, "
                            f"train perplexity: {perplexity(loss_cpu)}\t"
                            f"validation perplexity: {perplexity(validation_loss)}\tstep time: {step_time}")
                logger.info(f"eval time: {time.time()-eval_start_time}")

                if validation_loss < best_valid_loss:
                    best_valid_loss = validation_loss
                    logger.info(f"best valid loss achieved at epoch {epoch} step {i}")
                    best_epoch = epoch
                    best_step = i
                    # save model if achieve a new best valid loss
                    save_model(forward_deepnovo, backward_deepnovo, init_net)

                # back to train model
                forward_deepnovo.train()
                backward_deepnovo.train()

                start_time = time.time()
            # observed that most of gpu memory is unoccupied cache, so clear cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        #
        #print("train_losses",np.mean(train_losses),file=log_file_handle)
        #print("test_losses",np.mean(test_losses),file=log_file_handle)
        avg_train_loss.append(np.mean(train_losses))
        avg_test_loss.append(np.mean(test_losses))

    #print("Avg_train_loss",avg_train_loss,file=log_file_handle)
    #print("Avg_test_losses",avg_test_loss,file=log_file_handle)
    logger.info(f"best model at epoch {best_epoch} step {best_step}")
