from ..._creation import zeros, tensor
from ..._functional import cat


class PackedSequence:
    def __init__(self, data, batch_sizes, sorted_indices=None, unsorted_indices=None):
        self.data = data
        self.batch_sizes = batch_sizes
        self.sorted_indices = sorted_indices
        self.unsorted_indices = unsorted_indices

    def __iter__(self):
        yield self.data
        yield self.batch_sizes
        yield self.sorted_indices
        yield self.unsorted_indices


def pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True):
    if batch_first:
        input = input.transpose(0, 1)

    T, B = input.shape[0], input.shape[1]

    if isinstance(lengths, (list, tuple)):
        lengths_list = list(lengths)
    else:
        lengths_list = [int(lengths[i]) for i in range(lengths.shape[0])]

    if enforce_sorted:
        sorted_indices = None
        unsorted_indices = None
        sorted_lengths = lengths_list
        sorted_indices_list = list(range(B))
    else:
        indexed = sorted(enumerate(lengths_list), key=lambda x: -x[1])
        sorted_indices_list = [x[0] for x in indexed]
        sorted_lengths = [x[1] for x in indexed]

        unsorted_indices_list = [0] * B
        for new_idx, old_idx in enumerate(sorted_indices_list):
            unsorted_indices_list[old_idx] = new_idx

        sorted_indices = tensor(sorted_indices_list)
        unsorted_indices = tensor(unsorted_indices_list)

    # Compute batch_sizes: for each timestep, how many sequences are still active
    from ..._creation import tensor as _tensor
    batch_sizes_list = []
    for t in range(T):
        count = sum(1 for l in sorted_lengths if l > t)
        if count == 0:
            break
        batch_sizes_list.append(count)

    batch_sizes = _tensor(batch_sizes_list)

    # Pack data: concatenate valid elements at each timestep
    packed_parts = []
    for t in range(len(batch_sizes_list)):
        bs = batch_sizes_list[t]
        if enforce_sorted:
            for b in range(bs):
                packed_parts.append(input[t][b])
        else:
            for b in range(bs):
                packed_parts.append(input[t][sorted_indices_list[b]])

    if len(packed_parts) > 0:
        from ..._functional import stack
        packed_data = stack(packed_parts, dim=0)
    else:
        packed_data = input[:0].reshape(0, *input.shape[2:])

    return PackedSequence(packed_data, batch_sizes, sorted_indices, unsorted_indices)


def pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None):
    data = sequence.data
    batch_sizes = sequence.batch_sizes
    unsorted_indices = sequence.unsorted_indices

    if hasattr(batch_sizes, 'shape'):
        batch_sizes_list = [int(batch_sizes[i]) for i in range(batch_sizes.shape[0])]
    else:
        batch_sizes_list = list(batch_sizes)

    T = len(batch_sizes_list)
    B = batch_sizes_list[0] if T > 0 else 0

    if total_length is not None:
        T_out = total_length
    else:
        T_out = T

    feature_shape = data.shape[1:] if data.dim() > 1 else ()
    from ..._creation import tensor as _tensor
    from ..._dispatch import dispatch

    out_shape = (T_out, B) + feature_shape
    output = zeros(*out_shape, device=data.device, dtype=data.dtype)
    if padding_value != 0.0:
        output = output.fill_(padding_value)

    offset = 0
    for t in range(T):
        bs = batch_sizes_list[t]
        for b in range(bs):
            dispatch("setitem", output.device.type, output[t], b, data[offset + b])
        offset += bs

    lengths_list = [0] * B
    for t, bs in enumerate(batch_sizes_list):
        for b in range(bs):
            lengths_list[b] = t + 1
    lengths = _tensor(lengths_list)

    if unsorted_indices is not None:
        unsorted_list = [int(unsorted_indices[i]) for i in range(unsorted_indices.shape[0])]
        new_output = zeros(*out_shape, device=data.device, dtype=data.dtype)
        if padding_value != 0.0:
            new_output = new_output.fill_(padding_value)
        for orig_b in range(B):
            new_b = unsorted_list[orig_b]
            for t in range(T_out):
                dispatch("setitem", new_output.device.type, new_output[t], new_b, output[t][orig_b])
        output = new_output
        new_lengths = [0] * B
        for orig_b in range(B):
            new_lengths[unsorted_list[orig_b]] = lengths_list[orig_b]
        lengths = _tensor(new_lengths)

    if batch_first:
        output = output.transpose(0, 1)

    return output, lengths


def pack_sequence(sequences, enforce_sorted=True):
    lengths = [s.shape[0] for s in sequences]
    padded = pad_sequence(sequences, batch_first=False, padding_value=0.0)
    from ..._creation import tensor as _tensor
    return pack_padded_sequence(padded, _tensor(lengths), batch_first=False,
                                enforce_sorted=enforce_sorted)


def pad_sequence(sequences, batch_first=False, padding_value=0.0):
    max_len = max(s.shape[0] for s in sequences)
    B = len(sequences)
    feature_shape = sequences[0].shape[1:] if sequences[0].dim() > 1 else ()

    out_shape = (max_len, B) + feature_shape
    output = zeros(*out_shape, device=sequences[0].device, dtype=sequences[0].dtype)
    if padding_value != 0.0:
        output = output.fill_(padding_value)

    from ..._dispatch import dispatch
    for b, seq in enumerate(sequences):
        seq_len = seq.shape[0]
        for t in range(seq_len):
            dispatch("setitem", output.device.type, output[t], b, seq[t])

    if batch_first:
        output = output.transpose(0, 1)

    return output
