import mindspore

hip = None
cuda = mindspore.get_context('device_target') == 'GPU'
npu = mindspore.get_context('device_target') == 'Ascend'
