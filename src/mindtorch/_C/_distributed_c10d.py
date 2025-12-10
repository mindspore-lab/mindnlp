import pickle
from typing import List, Any
from datetime import timedelta
import mindspore

import mindtorch
from mindtorch import Tensor
from mindtorch.distributed import Store, TCPStore
from mindtorch.distributed.c10d import Backend, ReduceOp


class ProcessGroup:
    pass

class ProcessGroupGloo(Backend):
    def __init__(
        self,
        store: Store,
        rank: int,
        size: int,
        timeout: timedelta
    ) -> None:
        super().__init__(rank, size)
        self.store = store
        self.ranks = []
        self.pg = None

    def name(self) -> str:
        return 'gloo'

    def allreduce(self, tensors: List[Tensor], opts: Any) -> Any:
        if mindtorch.distributed.is_initialized():
            self._allreduce_new_pg(tensors[0], opts)
        else:
            self._allreduce_use_store(tensors, opts)

    def _allreduce_new_pg(self, tensor, opts):
        # Get all global ranks
        if len(self.ranks) == 0:
            rank_bytes = pickle.dumps(mindtorch.distributed.get_rank())
            self.store.set(f'__ar_rank_local_to_global_{self.rank_}', rank_bytes)
            for local_rank in range(self.size_):
                global_rank = pickle.loads(self.store.get(f'__ar_rank_local_to_global_{local_rank}'))
                self.ranks.append(global_rank)

        if self.pg is None:
            self.pg = mindtorch.distributed.new_group(self.ranks, backend='gloo')

        mindtorch.distributed.all_reduce(tensor, op=opts.reduceOp, group=self.pg, async_op=False)

    def _allreduce_use_store(self, tensors: List[Tensor], opts: Any) -> Any:
        tensor = tensors[0]
        tensor_bytes = pickle.dumps(tensor)
        self.store.set(f'__ar_data_{self.rank_}', tensor_bytes)

        # Gather all tensors
        gathered = []
        for i in range(self.size_):
            data = self.store.get(f'__ar_data_{i}')
            gathered.append(pickle.loads(data))
        stacked = mindtorch.stack(gathered)

        reduce_op = opts.reduceOp
        if reduce_op == ReduceOp.SUM:
            result = stacked.sum(dim=0)
        elif reduce_op == ReduceOp.MAX:
            if stacked.dtype == mindtorch.int32:
                result = stacked.to(mindtorch.int64).max(dim=0).values.to(mindtorch.int32)
            else:
                result = stacked.max(dim=0).values
        elif reduce_op == ReduceOp.MIN:
            if stacked.dtype == mindtorch.int32:
                result = stacked.to(mindtorch.int64).min(dim=0)[0].to(mindtorch.int32)
            else:
                result = stacked.min(dim=0)[0]
        elif reduce_op == ReduceOp.PRODUCT:
            result = stacked.prod(dim=0)
        else:
            raise ValueError(f'Unsupported reduce operation: {reduce_op}')

        tensors[0].copy_(result)
        self._synchronize_and_cleanup()

    def _synchronize_and_cleanup(self):
        if self.rank_ == 0:
            # Wait for the completion of allreduce() execution for other ranks and remove the tensor_i key
            # to prevent subsequent allreduce() exceptions.
            for i in range(1, self.size_):
                self.store.get(f'__ar_finish_1_{i}')
            for i in range(self.size_):
                self.store.delete_key(f'__ar_data_{i}')
                self.store.delete_key(f'__ar_finish_1_{i}')

            # Ensure that other ranks wait for the deletion of tensor_i key to complete.
            self.store.set('__ar_finish_all', '')

            # Ensure that rank 0 exits last to prevent errors in other ranks.
            for i in range(1, self.size_):
                self.store.get(f'__ar_finish_2_{i}')
                self.store.delete_key(f'__ar_finish_2_{i}')
            self.store.delete_key('__ar_finish_all')
        else:
            self.store.set(f'__ar_finish_1_{self.rank_}', '')
            self.store.get('__ar_finish_all')
            self.store.set(f'__ar_finish_2_{self.rank_}', '')

    def _set_sequence_number_for_group(self):
        pass


class ProcessGroupHCCL:
    def __init__(self, group_name):
        self.group_name = group_name

    def get_hccl_comm_name(self, global_rank):
        return self.group_name

    class Options: ...

def _resolve_process_group(group_name: str):
    mindspore.communication.destroy_group(group_name)
