# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Callback for timing.
"""
import time
from mindnlp.abc import Callback

class _Timer:
    """Timer."""

    def __init__(self, name):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()

    def start(self):
        """Start the timer."""
        assert not self.started_, f'{self.name_} timer has already been started'
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        """Stop the timer."""
        assert self.started_, f'{self.name_} timer is not started'
        self.elapsed_ += (time.time() - self.start_time)
        self.started_ = False

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False

    def elapsed(self, reset=True):
        """Calculate the elapsed time."""
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
        return elapsed_

class Timers:
    """Group of timers."""

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def __contains__(self, item):
        return item in self.timers

    def reset(self):
        """reset"""
        for timer in self.timers.values():
            timer.reset()

class TimerCallback(Callback):
    """
    Print relevant event information during the training process, such as
    training duration, evaluation duration, total duration.

    Args:
        print_steps (int): When to print time information.Default:-1.

            - -1: print once at the end of each epoch.
            - positive number n: print once n steps.

        time_ndigit (int): Number of decimal places to keep. Default:3
    """
    def __init__(self, print_steps=0, time_ndigit=3):
        assert isinstance(print_steps, int), "print_every must be an int number."
        self.timers = Timers()
        self.print_steps = print_steps
        self.time_ndigit = time_ndigit

    def train_begin(self, run_context):
        """
        Called once before the network training.

        Args:
            run_context (RunContext): Information about the model.

        """
        self.timers('total').start()
        self.timers('train').start()

    def train_end(self, run_context):
        """
        Called once after network training.

        Args:
            run_context (RunContext): Information about the model.

        """
        line = self.format_timer(train_end=True)
        print(f"Training finished{line}")

    def evaluate_begin(self, run_context):
        """
        Called once before the network evaluating.

        Args:
            run_context (RunContext): Information about the model.

        """
        self.timers('evaluate').start()

    def evaluate_end(self, run_context):
        """
        Called once after the network evaluating.

        Args:
            run_context (RunContext): Information about the model.

        """
        line = self.format_timer()
        print(f"Evaluating finished{line}")

    def train_step_begin(self, run_context):
        """
        Called before each train step beginning.

        Args:
            run_context (RunContext): Information about the model.

        """
        if self.print_steps > 0 and run_context.cur_step_nums % self.print_steps == 0:
            self.timers('step').start()

    def train_step_end(self, run_context):
        """
        Called after each train step finished.

        Args:
            run_context (RunContext): Information about the model.

        """
        if self.print_steps > 0 and run_context.cur_step_nums % self.print_steps == 0:
            line = self.format_timer()
            print(f"Running {run_context.cur_step_nums} batches{line}")

    def train_epoch_begin(self, run_context):
        """
        Called before each train epoch beginning.

        Args:
            run_context (RunContext): Information about the model.

        """
        if self.print_steps < 0:
            self.timers('epoch').start()

    def train_epoch_end(self, run_context):
        """
        Called after each train epoch finished.

        Args:
            run_context (RunContext): Information about the model.

        """
        if self.print_steps < 0 and run_context.cur_epoch_nums % abs(self.print_steps) == 0:
            line = self.format_timer()
            print(f"Running {run_context.cur_epoch_nums} epochs{line}")

    def format_timer(self, reset=True, train_end=False):
        """
        Format the output.

        Args:
            run_context (RunContext): Information about the model.

        """
        line = ''
        timers = ['step', 'epoch', 'evaluate', 'train', 'total']
        for timer_name in timers:
            if train_end is False:
                if not timer_name in self.timers or timer_name == 'train' or timer_name == 'total':
                    continue
            timer = self.timers(timer_name)
            elapsed = round(timer.elapsed(reset=reset), self.time_ndigit)
            if elapsed != 0:
                line = line + f', {timer_name} time cost: {elapsed}s'
        return line
