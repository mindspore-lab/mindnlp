Use callback
=============

Callback is a module closely related to Trainer.
Using callback in Trainer can realize timing,
early stop, saving checkpoint and other operations
required in model training.
At the same time, MindNLP also supports custom callback function.

Use Callback in Engine
^^^^^^^^^^^^^^^^^^^^^^^^
Callback needs to be used on the premise that trainer or evaluator
has been defined. MindNLP supports passing in two types of parameters
to the enigne: Callback and Callback List.
Engine will automatically execute the operations specified by these callbacks.

The code of using engine is as follows:

.. code:: python

    import mindspore.dataset as ds

    from mindspore import nn

    from mindnlp.engine.trainer import Trainer
    from mindnlp.engine.callbacks.earlystop_callback import EarlyStopCallback

    class MyDataset:
    """Dataset"""
    def __init__(self):
        self.data = np.random.randn(20, 3).astype(np.float32)
        self.label = list(np.random.choice([0, 1]).astype(np.float32) for i in range(20))
        self.length = list(np.random.choice([0, 1]).astype(np.float32) for i in range(20))
    def __getitem__(self, index):
        return self.data[index], self.label[index], self.length[index]
    def __len__(self):
        return len(self.data)

    class MyModel(nn.Cell):
        """Model"""
        def __init__(self):
            super().__init__()
            self.fc = nn.Dense(3, 1)
        def construct(self, data):
            output = self.fc(data)
            return output

    # Define Dataset
    dataset_generator = MyDataset()
    train_dataset = ds.GeneratorDataset(dataset_generator, ["data", "label", "length"], shuffle=False)
    eval_dataset = ds.GeneratorDataset(dataset_generator, ["data", "label", "length"], shuffle=False)
    train_dataset = train_dataset.batch(4)
    eval_dataset = eval_dataset.batch(4)
    # Define Model
    net = MyModel()
    net.update_parameters_name('net.')
    # Define Loss function
    loss_fn = nn.MSELoss()
    # Define Optimizer
    optimizer = nn.Adam(net.trainable_params(), learning_rate=0.01)
    # Define Callback
    timer_callback = TimerCallback(print_steps=2)
    # Define Trainer
    trainer = Trainer(network=net, train_dataset=train_dataset, eval_dataset=eval_dataset,
                      epochs=6, optimizer=optimizer, loss_fn=loss_fn, callbacks=timer_callback)
    # Run Trainer
    trainer.run(tgt_columns='label', jit=True)

Callbacks in MindNLP
^^^^^^^^^^^^^^^^^^^^^^^^
MindNLP provides many common callbacks, such as ``TimerCallback``,
``EarlyStopCallback``, ``BestModelCallback`` and so on.
For specific Callback, please refer to :py:class:`~mindnlp.engine.callbacks`

.. code:: python

    from mindnlp.engine.callbacks import TimerCallback, EarlyStopCallback, BestModelCallback, CheckpointCallback

    callbacks = [
        TimerCallback(print_steps=2),
        EarlyStopCallback(patience=2),
        BestModelCallback(save_path='save/callback/best_model', auto_load=True),
        CheckpointCallback(save_path='save/callback/ckpt_files', epochs=2,
                           keep_checkpoint_max=2)
    ]

Custom Callback
^^^^^^^^^^^^^^^^^^^^^^^
Here we take a simple Callback as an example,
its function is to print the average training loss of each Epoch.

Create Callback
----------------
To customize Callback, we need to implement a class that
inherits from Callback. Here we define MyCallBack,
which inherits from :py:class:`~mindnlp.abc.callback`.

Specifies the phase of calling the Callback
--------------------------------------------
All class methods in Callback
will be called at a specific stage during Trainer's training.
For example, train_begin() will be called at the beginning of training,
and epoch_end() will be called at the end of each epoch.
For specific class methods, see the ``Callback`` documentation.
Here, MyCallBack calls epoch_end() at the end of each epoch,
output the loss of the current epoch.

Access the internal information of Engine
------------------------------------------
All methods in Callback contain parameter ``run_context``
that can access the internal information of the Engine, such as
current numbers of steps, current numbers of epochs, loss value, etc.
Here, MyCallBack needs to get the current number of epochs of the Trainer
and the average loss value after each epoch.

.. code:: python

    from mindspore import logging
    from mindnlp.abc import Callback

    class MyCallBack(Callback):
        def __init__(self):
            self.epoch = run_context.cur_epoch_nums
            self.loss = 0

        def epoch_end(self, run_context):
            self.loss = run_context.loss
            logging.info('Avg loss at epoch %d, %.6f', self.epoch, avg_loss)

    my_callback = MyCallBack()
    trainer = Trainer(network=net, train_dataset=train_dataset, eval_dataset=eval_dataset,
                      epochs=6, optimizer=optimizer, loss_fn=loss_fn, callbacks=my_callback)
    trainer.run(tgt_columns='label', jit=True)
