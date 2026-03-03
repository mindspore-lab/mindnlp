"""ProcessGroupGloo: Pure Python TCP-based collective communication backend."""

import numpy as np

from .._work import Work
from .._process_group import ProcessGroup
from ._tcp_transport import TcpTransport
from ._numpy_collectives import apply_reduce_op, serialize_array, deserialize_array
from .._reduce_op import RedOpType


class ProcessGroupGloo(ProcessGroup):
    """Gloo-compatible process group using pure Python TCP transport.

    All collectives are synchronous (gather-to-root + broadcast-from-root).
    Work objects are returned already completed.
    """

    def __init__(self, store, rank, size, group_name="", group_ranks=None):
        super().__init__(rank, size)
        self._group_name = group_name
        self._ranks = group_ranks
        self._store = store
        self._transport = TcpTransport(store, rank, size, prefix=group_name)

    def _tensor_to_numpy(self, tensor):
        """Convert a CPU tensor to a contiguous numpy array."""
        arr = tensor._numpy_view()
        return np.ascontiguousarray(arr)

    def _write_numpy_to_tensor(self, arr, tensor):
        """Write numpy array data back into a tensor in-place."""
        dst = tensor._numpy_view()
        np.copyto(dst, arr.reshape(dst.shape))

    def _make_work(self):
        """Return an already-completed Work object (synchronous collectives)."""
        work = Work(stream=None)
        work._completed = True
        return work

    def allreduce(self, tensor, op=0):
        """All-reduce: all ranks reduce and receive the result."""
        if self._size == 1:
            return self._make_work()

        arr = self._tensor_to_numpy(tensor)
        serialized = serialize_array(arr)

        # All non-root send to rank 0
        if self._rank != 0:
            self._transport.send_to(0, serialized)
        else:
            # Rank 0 receives from all peers and reduces
            result = arr.copy()
            for peer in range(1, self._size):
                peer_data = self._transport.recv_from(peer)
                peer_arr = deserialize_array(peer_data)
                result = apply_reduce_op(op, result, peer_arr)

            # Handle AVG: divide by world_size
            if int(op) == int(RedOpType.AVG):
                result = result / self._size

            # Rank 0 broadcasts result to all
            result_serialized = serialize_array(result)
            for peer in range(1, self._size):
                self._transport.send_to(peer, result_serialized)

            # Write result back into tensor
            self._write_numpy_to_tensor(result, tensor)

        # Non-root ranks receive the result
        if self._rank != 0:
            result_data = self._transport.recv_from(0)
            result = deserialize_array(result_data)
            self._write_numpy_to_tensor(result, tensor)

        return self._make_work()

    def broadcast(self, tensor, root=0):
        """Broadcast: root sends tensor to all other ranks."""
        if self._size == 1:
            return self._make_work()

        if self._rank == root:
            arr = self._tensor_to_numpy(tensor)
            serialized = serialize_array(arr)
            for peer in range(self._size):
                if peer != root:
                    self._transport.send_to(peer, serialized)
        else:
            data = self._transport.recv_from(root)
            arr = deserialize_array(data)
            self._write_numpy_to_tensor(arr, tensor)

        return self._make_work()

    def allgather(self, output_tensor, input_tensor):
        """All-gather: gather input from all ranks, concatenate into output."""
        if self._size == 1:
            # Just copy input to output
            np.copyto(output_tensor._numpy_view(), input_tensor._numpy_view())
            return self._make_work()

        arr = self._tensor_to_numpy(input_tensor)
        serialized = serialize_array(arr)

        # All ranks send to rank 0
        if self._rank != 0:
            self._transport.send_to(0, serialized)
        else:
            # Rank 0 gathers from all
            gathered = [arr]
            for peer in range(1, self._size):
                peer_data = self._transport.recv_from(peer)
                peer_arr = deserialize_array(peer_data)
                gathered.append(peer_arr)

            # Concatenate along first dimension
            result = np.concatenate(gathered, axis=0)
            result_serialized = serialize_array(result)

            # Rank 0 broadcasts concatenated result
            for peer in range(1, self._size):
                self._transport.send_to(peer, result_serialized)

            self._write_numpy_to_tensor(result, output_tensor)

        # Non-root ranks receive the result
        if self._rank != 0:
            result_data = self._transport.recv_from(0)
            result = deserialize_array(result_data)
            self._write_numpy_to_tensor(result, output_tensor)

        return self._make_work()

    def reduce(self, tensor, dst=0, op=0):
        """Reduce: all ranks send to dst, dst receives reduced result."""
        if self._size == 1:
            return self._make_work()

        arr = self._tensor_to_numpy(tensor)
        serialized = serialize_array(arr)

        # All non-dst send to dst
        if self._rank != dst:
            self._transport.send_to(dst, serialized)
        else:
            # dst receives from all and reduces
            result = arr.copy()
            for peer in range(self._size):
                if peer != dst:
                    peer_data = self._transport.recv_from(peer)
                    peer_arr = deserialize_array(peer_data)
                    result = apply_reduce_op(op, result, peer_arr)

            # Handle AVG
            if int(op) == int(RedOpType.AVG):
                result = result / self._size

            self._write_numpy_to_tensor(result, tensor)

        return self._make_work()

    def reduce_scatter(self, output_tensor, input_tensor, op=0):
        """Reduce-scatter: reduce input, split into chunks, scatter to ranks."""
        if self._size == 1:
            np.copyto(output_tensor._numpy_view(), input_tensor._numpy_view())
            return self._make_work()

        arr = self._tensor_to_numpy(input_tensor)
        serialized = serialize_array(arr)

        # All ranks send to rank 0
        if self._rank != 0:
            self._transport.send_to(0, serialized)
        else:
            # Rank 0 reduces
            result = arr.copy()
            for peer in range(1, self._size):
                peer_data = self._transport.recv_from(peer)
                peer_arr = deserialize_array(peer_data)
                result = apply_reduce_op(op, result, peer_arr)

            # Handle AVG
            if int(op) == int(RedOpType.AVG):
                result = result / self._size

            # Split into chunks (split along first dimension)
            chunk_size = result.shape[0] // self._size
            chunks = np.split(result, self._size, axis=0)

            # Send chunk i to rank i
            for peer in range(self._size):
                chunk_serialized = serialize_array(chunks[peer])
                if peer == 0:
                    self._write_numpy_to_tensor(chunks[peer], output_tensor)
                else:
                    self._transport.send_to(peer, chunk_serialized)

        # Non-root ranks receive their chunk
        if self._rank != 0:
            chunk_data = self._transport.recv_from(0)
            chunk = deserialize_array(chunk_data)
            self._write_numpy_to_tensor(chunk, output_tensor)

        return self._make_work()

    def scatter(self, output_tensor, input_tensor, src=0):
        """Scatter: src splits input into chunks, sends chunk i to rank i."""
        if self._size == 1:
            np.copyto(output_tensor._numpy_view(), input_tensor._numpy_view())
            return self._make_work()

        if self._rank == src:
            arr = self._tensor_to_numpy(input_tensor)
            # Split into chunks
            chunk_size = arr.shape[0] // self._size
            chunks = np.split(arr, self._size, axis=0)

            # Send chunk i to rank i
            for peer in range(self._size):
                chunk_serialized = serialize_array(chunks[peer])
                if peer == src:
                    self._write_numpy_to_tensor(chunks[peer], output_tensor)
                else:
                    self._transport.send_to(peer, chunk_serialized)
        else:
            # Receive chunk from src
            chunk_data = self._transport.recv_from(src)
            chunk = deserialize_array(chunk_data)
            self._write_numpy_to_tensor(chunk, output_tensor)

        return self._make_work()

    def barrier(self):
        """Barrier: all ranks synchronize."""
        if self._size == 1:
            return self._make_work()

        # All ranks send a 1-byte token to rank 0
        if self._rank != 0:
            self._transport.send_to(0, b"\x00")
        else:
            # Rank 0 waits for all
            for peer in range(1, self._size):
                self._transport.recv_from(peer)
            # Rank 0 sends ack to all
            for peer in range(1, self._size):
                self._transport.send_to(peer, b"\x00")

        # Non-root ranks wait for ack
        if self._rank != 0:
            self._transport.recv_from(0)

        return self._make_work()

    def send(self, tensor, dst):
        """Point-to-point send."""
        arr = self._tensor_to_numpy(tensor)
        serialized = serialize_array(arr)
        self._transport.send_to(dst, serialized)
        return self._make_work()

    def recv(self, tensor, src):
        """Point-to-point receive."""
        data = self._transport.recv_from(src)
        arr = deserialize_array(data)
        self._write_numpy_to_tensor(arr, tensor)
        work = self._make_work()
        work._source_rank = src
        return work

    def all_to_all(self, output_tensors, input_tensors):
        """All-to-all: each rank sends tensor i to rank i, receives from all.

        Uses P2P send/recv with deadlock-free ordering:
        - For peer < rank: recv first, then send
        - For peer > rank: send first, then recv
        - For peer == rank: local copy
        """
        if self._size == 1:
            np.copyto(output_tensors[0]._numpy_view(), input_tensors[0]._numpy_view())
            return self._make_work()

        # Local copy first (rank sends to itself)
        np.copyto(output_tensors[self._rank]._numpy_view(),
                  input_tensors[self._rank]._numpy_view())

        # P2P exchange with deadlock-free ordering
        for peer in range(self._size):
            if peer == self._rank:
                continue

            if self._rank < peer:
                # Lower rank sends first, then receives
                arr = self._tensor_to_numpy(input_tensors[peer])
                serialized = serialize_array(arr)
                self._transport.send_to(peer, serialized)

                data = self._transport.recv_from(peer)
                arr = deserialize_array(data)
                self._write_numpy_to_tensor(arr, output_tensors[peer])
            else:
                # Higher rank receives first, then sends
                data = self._transport.recv_from(peer)
                arr = deserialize_array(data)
                self._write_numpy_to_tensor(arr, output_tensors[peer])

                arr = self._tensor_to_numpy(input_tensors[peer])
                serialized = serialize_array(arr)
                self._transport.send_to(peer, serialized)

        return self._make_work()

    def destroy(self):
        """Clean up transport resources."""
        self._transport.close()
