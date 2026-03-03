class P2POp:
    def __init__(self, op, tensor, peer=None, group=None, tag=0, group_peer=None):
        self.op = op
        self.tensor = tensor
        self.peer = peer
        self.group = group
        self.tag = tag
        self.group_peer = group_peer


def batch_isend_irecv(p2p_op_list):
    works = []
    for p2p in p2p_op_list:
        dst_or_src = p2p.peer
        kwargs = {"group": p2p.group, "tag": p2p.tag}
        # isend/irecv accept group_dst/group_src as keyword args
        if dst_or_src is None and p2p.group_peer is not None:
            # Determine the right keyword based on the op name
            op_name = getattr(p2p.op, "__name__", "")
            if "send" in op_name:
                kwargs["group_dst"] = p2p.group_peer
            else:
                kwargs["group_src"] = p2p.group_peer
        work = p2p.op(p2p.tensor, dst_or_src, **kwargs)
        works.append(work)
    return works
