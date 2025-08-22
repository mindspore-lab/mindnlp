from mindnlp import core

def InfNanRemoveLogitsProcessor_call(self, input_ids, scores):
    # set all +/-inf values to max/min possible value
    scores_processed = scores
    scores_processed = core.where(scores == float("inf"), core.finfo(scores.dtype).max, scores_processed)
    scores_processed = core.where(scores == -float("inf"), core.finfo(scores.dtype).min, scores_processed)
    # set all nan values to 0.0
    scores_processed = core.where(scores != scores, 0.0, scores)

    return scores_processed
