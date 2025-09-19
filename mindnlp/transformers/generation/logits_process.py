import mindtorch

def InfNanRemoveLogitsProcessor_call(self, input_ids, scores):
    # set all +/-inf values to max/min possible value
    scores_processed = scores
    scores_processed = mindtorch.where(scores == float("inf"), mindtorch.finfo(scores.dtype).max, scores_processed)
    scores_processed = mindtorch.where(scores == -float("inf"), mindtorch.finfo(scores.dtype).min, scores_processed)
    # set all nan values to 0.0
    scores_processed = mindtorch.where(scores != scores, 0.0, scores)

    return scores_processed
