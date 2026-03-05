/*
 * AscendC AddCustom kernel — element-wise vector addition on NPU AI cores.
 *
 * Entry point: add_custom(x, y, z, workspace, tiling)
 *   - x, y   : input fp16 tensors  (GM_ADDR)
 *   - z      : output fp16 tensor  (GM_ADDR)
 *   - workspace : unused (GM_ADDR, required by AscendC calling convention)
 *   - tiling : device buffer holding a single uint32_t = totalLength
 *
 * Compile target: ascend910b (Ascend 910B)
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int BUFFER_NUM = 2;

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t totalLength) {
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = blockLength / TILE_LENGTH;
        xGm.SetGlobalBuffer(
            (__gm__ half *)x + this->blockLength * GetBlockIdx(),
            this->blockLength);
        yGm.SetGlobalBuffer(
            (__gm__ half *)y + this->blockLength * GetBlockIdx(),
            this->blockLength);
        zGm.SetGlobalBuffer(
            (__gm__ half *)z + this->blockLength * GetBlockIdx(),
            this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(half));
    }

    __aicore__ inline void Process() {
        for (int32_t i = 0; i < this->tileNum; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * TILE_LENGTH], TILE_LENGTH);
        DataCopy(yLocal, yGm[progress * TILE_LENGTH], TILE_LENGTH);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        Add(zLocal, xLocal, yLocal, TILE_LENGTH);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        DataCopy(zGm[progress * TILE_LENGTH], zLocal, TILE_LENGTH);
        outQueueZ.FreeTensor(zLocal);
    }

    static constexpr int TILE_LENGTH = 128;
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<half> xGm, yGm, zGm;
    uint32_t blockLength = 0;
    uint32_t tileNum = 0;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y,
                                                  GM_ADDR z,
                                                  GM_ADDR workspace,
                                                  GM_ADDR tiling) {
    KernelAdd op;
    uint32_t totalLength = *((__gm__ uint32_t *)tiling);
    op.Init(x, y, z, totalLength);
    op.Process();
}
