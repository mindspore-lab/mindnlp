import glob
import logging
import os
import random
import re
from typing import *

import numpy as np
import mindspore
import tqdm


# todo: fix logging in this file


def get_save_dir(base_dir, name):
    """
    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.

    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    base_path = os.path.join(base_dir, name)
    match_dirs = sorted(glob.glob(base_path + "*"))

    if len(match_dirs) == 0:  # none existed
        unused_dir = base_path + "-01"
    else:       # increment from the last one
        last = match_dirs[-1]
        r = re.match(".*-(\d\d)", last)
        next_num = int(r.group(1)) + 1

        unused_dir = base_path + f'-{next_num:02d}'

    # log.info(f'Will create new save_dir: {unused_dir}')
    os.makedirs(unused_dir)
    return unused_dir


def visualize(tbx, pred_dict: Union[Dict, List], step, split, num_visuals):
    """Visualize text examples to TensorBoard.

    Args:
        tbx (tensorboardX.SummaryWriter): Summary writer.
    """
    if num_visuals <= 0:
        return
    if num_visuals > len(pred_dict):
        num_visuals = len(pred_dict)

    for i in range(num_visuals):
        # unpack tuple
        orig_input, orig_target, actual_output = pred_dict[i]

        tbl_fmt = (f'- **Source:** {orig_input}\n'
                   + f'- **Target:** {orig_target}\n'
                   + f'- **Predicted:** {actual_output}\n')
        tbx.add_text(tag=f'{split}/{i+1}_of_{num_visuals}',
                     text_string=tbl_fmt,
                     global_step=step)


def save_preds(preds: List[Tuple[str,str,str]], save_dir, file_name='predictions.csv'):
    """Save predictions `preds` to a CSV file named `file_name` in `save_dir`.

    Args:
        preds (list): List of predictions each of the form (source, target, actual),
        save_dir (str): Directory in which to save the predictions file.
        file_name (str): File name for the CSV file.

    Returns:
        save_path (str): Path where CSV file was saved.
    """
    save_path = os.path.join(save_dir, file_name)
    np.savetxt(save_path, np.array(preds), delimiter='|', fmt='%s')

    return save_path

# This function is like a rouge metric
def masked_token_match(tgt_ids: mindspore.Tensor, outputs: mindspore.Tensor,
                       return_indices=False) -> Union[Tuple[int,int], Tuple[int, int, mindspore.Tensor]]:
    """
    Takes generated outputs and tgt_ids, both of size (batch_size, seq_len), where seq_len may differ
    For all tokens in tgt_ids that are not PAD or EOS,
        - check that they are equal
        - count all the examples that are an exact match

    Returns:
        - total_matches_no_eos: all the matches where we get everything except EOS correct
        - total_matches_with_eos: all matches where we get everything including EOS
        - optional (if return_indices): the indices where we have a match on everything up to the EOS token

    """
    # left-shift
    # assert (output_ids[:,0] == 0)       # T5 should start with a pad token; other models could vary
    output_shifted = outputs[:,1:]

    if output_shifted.shape <= tgt_ids.shape:
        # create output_padded, which truncates output at tgt_ids size, filling with pad tokens
        output_padded = mindspore.ops.zeros_like(tgt_ids)
        output_padded[:output_shifted.shape[0], :output_shifted.shape[1]] = output_shifted
    else:       # output_shifted is bigger
        # so copy only up to the target IDs length
        output_padded = output_shifted[:,:tgt_ids.shape[1]]     # copy all rows (bs) and up to tgt_ids length

    # compare where tokens are > 1 (i.e. not pad or EOS)
    match_indices = output_padded == tgt_ids          # either they match
    matches_no_eos = mindspore.Tensor.logical_or(match_indices, tgt_ids < 2)   # or we ignore them (pad and eos)
    matches_with_eos = mindspore.Tensor.logical_or(match_indices, tgt_ids < 1) # or we ignore them (just pad)
    total_matches_no_eos = mindspore.sum(mindspore.ops.all(matches_no_eos, axis=1))
    total_matches_with_eos = mindspore.sum(mindspore.ops.all(matches_with_eos, axis=1))

    correct_indices = mindspore.ops.nonzero(mindspore.ops.all(matches_no_eos, axis=1))

    if return_indices:
        return total_matches_no_eos, total_matches_with_eos, correct_indices
    else:
        return total_matches_no_eos, total_matches_with_eos


# We use this for evaluation
class AverageMeter:
    """Keep track of average values over time.

    Adapted from:
        > https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset meter."""
        self.__init__()

    def update(self, val, num_samples=1):
        """Update meter with new value `val`, the average of `num` samples.

        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    mindspore.dataset.config.set_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)


# def get_available_devices():
#     """Get IDs of all available GPUs.

#     Returns:
#         device (torch.device): Main device (GPU 0 or CPU).
#         gpu_ids (list): List of IDs of all GPUs that are available.
#     """
#     gpu_ids = []
#     if torch.cuda.is_available():
#         gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
#         device = torch.device(f'cuda:{gpu_ids[0]}')
#         torch.cuda.set_device(device)
#     else:
#         device = torch.device('cpu')

#     return device, gpu_ids

def get_logger(log_dir, name, log_level="debug"):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.

    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.

    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """
    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.

        See Also:
            > https://stackoverflow.com/questions/38543506
        """
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    if log_level == "debug":
        logger.setLevel(logging.DEBUG)
    elif log_level == "info":
        logger.setLevel(logging.INFO)
    else:
        raise ValueError(f"Invalid log level {log_level}")

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    # console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
    #                                       datefmt='%m.%d.%y %H:%M:%S')
    console_formatter = logging.Formatter(
        '[%(asctime)s] [%(filename)s:%(lineno)s - %(funcName)s()] %(message)s',
        datefmt='%m.%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger