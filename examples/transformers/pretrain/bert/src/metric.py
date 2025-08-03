import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

from mindspore import Tensor
import numpy as np

from icecream import ic

def metric_fn(masked_lm_loss, masked_lm_log_probs, masked_lm_ids,
              masked_lm_weights, masked_lm_positions, next_sentence_loss,
              next_sentence_log_probs, next_sentence_labels):
    """Computes the loss and accuracy of the model."""

    # compute MLM Accuracy
    bs = masked_lm_positions.shape[0]
    mask_nums = masked_lm_positions.shape[1]
    mlm_predictions = masked_lm_log_probs
    for idx in range(bs):
        mlm_prediction = masked_lm_log_probs[idx][masked_lm_positions[idx]]
        if idx == 0:
            mlm_predictions = mlm_prediction
        else:
            mlm_predictions = ops.Concat()((mlm_predictions, mlm_prediction))

    mlm_predictions = mlm_predictions.reshape(bs, mask_nums, -1)
    mlm_predictions = mlm_predictions.transpose(0, 2, 1)

    mlm_predictions = mlm_predictions
    masked_lm_ids = masked_lm_ids
    mlm_correct = (mlm_predictions.argmax(1) == masked_lm_ids).astype(np.float32)
    masked_lm_accuracy = mlm_correct.sum() / len(mlm_correct)

    masked_lm_mean_loss = (masked_lm_loss * masked_lm_weights).mean()

    # compute NSP Accuracy
    next_sentence_labels = next_sentence_labels.reshape(-1,)
    next_sentence_log_probs = next_sentence_log_probs
    next_sentence_labels = next_sentence_labels
    nsp_correct = (next_sentence_log_probs.argmax(1) == next_sentence_labels).astype(np.float32)
    next_sentence_accuracy = nsp_correct.sum() / len(nsp_correct)
    next_sentence_mean_loss = next_sentence_loss.mean()

    eval_result = {
        "masked_lm_accuracy": masked_lm_accuracy,
        "masked_lm_loss": masked_lm_mean_loss,
        "next_sentence_accuracy": next_sentence_accuracy,
        "next_sentence_loss": next_sentence_mean_loss,
    }
    return eval_result

    
