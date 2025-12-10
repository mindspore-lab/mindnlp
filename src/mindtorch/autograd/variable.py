#!/usr/bin/env python
# -*- coding: utf-8 -*-
from mindtorch import Tensor
import logging

class Variable(Tensor):
    def __new__(cls, data, requires_grad=None, volatile=None):
        logging.warning("The Variable API has been deprecated, use Tensor instead.")
        obj = Tensor.__new__(cls)
        return obj

    def __init__(self, data, requires_grad=None, volatile=None):
        if volatile:
            logging.warning("UserWarning:volatile was removed (Variable.volatile is always False), "
                    "please use with mindtorch.no_grad() instead.")
        Tensor.__init__(self, data, requires_grad=requires_grad)
