"""Weight initialization stubs."""


def uniform_(tensor, a=0.0, b=1.0):
    raise NotImplementedError("uniform_ is not yet implemented")


def normal_(tensor, mean=0.0, std=1.0):
    raise NotImplementedError("normal_ is not yet implemented")


def constant_(tensor, val):
    raise NotImplementedError("constant_ is not yet implemented")


def ones_(tensor):
    raise NotImplementedError("ones_ is not yet implemented")


def zeros_(tensor):
    raise NotImplementedError("zeros_ is not yet implemented")


def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    raise NotImplementedError("kaiming_uniform_ is not yet implemented")


def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    raise NotImplementedError("kaiming_normal_ is not yet implemented")


def xavier_uniform_(tensor, gain=1.0):
    raise NotImplementedError("xavier_uniform_ is not yet implemented")


def xavier_normal_(tensor, gain=1.0):
    raise NotImplementedError("xavier_normal_ is not yet implemented")


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    raise NotImplementedError("trunc_normal_ is not yet implemented")
