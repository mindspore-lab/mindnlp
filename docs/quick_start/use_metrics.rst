Use Metrics
============

There are a variety of
:py:class:`~mindnlp.abc.Metric`
in MindNLP for model evaluation:
:py:class:`~mindnlp.engine.metrics.accuracy.Accuracy`,
:py:class:`~mindnlp.engine.metrics.bleu.BleuScore`,
:py:class:`~mindnlp.engine.metrics.confusion_matrix.ConfusionMatrix`,
:py:class:`~mindnlp.engine.metrics.distinct.Distinct`,
:py:class:`~mindnlp.engine.metrics.em_score.EmScore`,
:py:class:`~mindnlp.engine.metrics.f1.F1Score`,
:py:class:`~mindnlp.engine.metrics.matthews.MatthewsCorrelation`,
:py:class:`~mindnlp.engine.metrics.pearson.PearsonCorrelation`,
:py:class:`~mindnlp.engine.metrics.perplexity.Perplexity`,
:py:class:`~mindnlp.engine.metrics.precision.Precision`,
:py:class:`~mindnlp.engine.metrics.recall.Recall`,
:py:class:`~mindnlp.engine.metrics.rouge.RougeL`,
:py:class:`~mindnlp.engine.metrics.rouge.RougeN`,
:py:class:`~mindnlp.engine.metrics.spearman.SpearmanCorrelation`.

We can use these pre-defined metrics directly, by instantiating
some of the classes and passing the instantiated objects into
:py:class:`~mindnlp.engine.trainer.Trainer` as one of its
parameters.

Taking the use of
:py:class:`~mindnlp.engine.metrics.accuracy.Accuracy`
as an example, the code of using metrics for model training and
evaluation is as follows:

.. code-block:: python

    from mindnlp.engine.metrics import Accuracy

    from mindnlp.engine.trainer import Trainer

    metric = Accuracy()

    trainer = Trainer(network=network, train_dataset=train_dataset, eval_dataset=eval_dataset,
                        metrics=metric, epochs=epochs, loss_fn=loss_fn, optimizer=optimizer,
                        callbacks=callbacks)

    trainer.run(tgt_columns="label", jit=False)

Define a New Metric
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the metric we need is not provided by MindNLP, it is
still simple and easy for us to define our own metric.

All of the classes of metrics defined in MindNLP are
inherited from the base class
:py:class:`~mindnlp.abc.Metric`.
When defining our own metric class, it is also necessary
to extend :py:class:`~mindnlp.abc.Metric`
and rewrite the functions of it:

* ``__init__()``: initializes the metric.
* ``clear()``: clears the internal evaluation results.
* ``eval()``: computes and returns the value of the metric.
* ``update(*inputs)``: updates the local variables.
* ``get_metric_name()``: returns the name of the metric.

After finishing those operations, the steps to train and evaluate models
using self-defined metrics are the same as mentioned above.
