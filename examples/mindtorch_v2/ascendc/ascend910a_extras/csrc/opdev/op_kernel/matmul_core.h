#ifndef _MATMUL_CORE_H
#define _MATMUL_CORE_H

#include "kernel_operator.h"

// FIXME: must be n % 128 == 0 and k % 128 == 0
template<typename scalar_t, typename acc_t, typename id_t>
class MatMulNT {
public:
    static constexpr int BLOCK_M = 64;
    static constexpr int BLOCK_K = 64;
    static constexpr int BLOCK_N = 64;

    static constexpr int L1_STAGE = 1;
    // static constexpr int L1_STAGE = 2;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::A1, L1_STAGE> a1_que;
    AscendC::TQue<AscendC::TPosition::A2, 1> a2_que;
    AscendC::TQue<AscendC::TPosition::B1, L1_STAGE> b1_que;
    AscendC::TQue<AscendC::TPosition::B2, 1> b2_que;
    AscendC::TQue<AscendC::TPosition::CO1, 1> co1_que;
    AscendC::TQue<AscendC::TPosition::CO2, 1> co2_que;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> c_que;

    AscendC::GlobalTensor<scalar_t> a_gm;
    AscendC::GlobalTensor<scalar_t> b_gm; // transposed
    AscendC::GlobalTensor<scalar_t> c_gm;

    int m, n, k;
    uint16_t curr_block_m;
    __gm__ scalar_t* a;
    __gm__ scalar_t* b;
    __gm__ scalar_t* c;

    __aicore__ inline MatMulNT() {
        m = 0;
        n = 0;
        k = 0;
        curr_block_m = 0;
    }
    __aicore__ inline void Process() {
        for (int mi = 0; mi < m; mi += BLOCK_M) {
            for (int ni = 0; ni < n; ni += BLOCK_N) {
                this->curr_block_m = (m - mi < BLOCK_M) ? (m - mi) : BLOCK_M;
                c_gm.SetGlobalBuffer(c + mi * n + ni);
                AscendC::LocalTensor<acc_t> acc = co1_que.AllocTensor<acc_t>();
                // for (int ki = 0; ki < k; ki += BLOCK_K) {
                for (int ki = 0; ki < k; ki += BLOCK_K * L1_STAGE) {
                    a_gm.SetGlobalBuffer(a + mi * k + ki);
                    b_gm.SetGlobalBuffer(b + ni * k + ki);
                    CopyGmToL2();
                    CopyL2ToL1();
                    Mma(acc, ki == 0);
                    // for (int s = 0; s < L1_STAGE; ++s) {
                    // // for (int s = 1; s < L1_STAGE; ++s) {
                    // // for (int s = 0; s < 1; ++s) {
                    //     CopyGmToL2Staged(s);
                    //     CopyL2ToL1();
                    //     Mma(acc, ki == 0);
                    // }
                }
                co1_que.EnQue(acc);
                CopyCO1ToCO2();
                CopyCO2ToGm();
            }
        }
    }

    __aicore__ inline void InitPipe() {
        pipe.InitBuffer(a1_que, L1_STAGE, BLOCK_M * BLOCK_K * sizeof(scalar_t));
        pipe.InitBuffer(a2_que, 1, BLOCK_M * BLOCK_K * sizeof(scalar_t));
        pipe.InitBuffer(b1_que, L1_STAGE, BLOCK_K * BLOCK_N * sizeof(scalar_t));
        pipe.InitBuffer(b2_que, 1, BLOCK_K * BLOCK_N * sizeof(scalar_t));
        pipe.InitBuffer(co1_que, 1, BLOCK_M * BLOCK_N * sizeof(acc_t));
        pipe.InitBuffer(co2_que, 1, BLOCK_M * BLOCK_N * sizeof(acc_t));
        pipe.InitBuffer(c_que, 1, BLOCK_M * BLOCK_N * sizeof(scalar_t));
    }

    __aicore__ inline void InitSize(int m, int n, int k) {
        this->m = m;
        this->n = n;
        this->k = k;
    }

    __aicore__ inline void InitBuffer(
        __gm__ scalar_t* a,
        __gm__ scalar_t* b,
        __gm__ scalar_t* c
    ) {
        this->a = a;
        this->b = b;
        this->c = c;
    }

    __aicore__ inline void CopyGmToL2() {
        AscendC::LocalTensor<scalar_t> a1 = a1_que.AllocTensor<scalar_t>();
        AscendC::LocalTensor<scalar_t> b1 = b1_que.AllocTensor<scalar_t>();

        // for a: nd -> nz
        for (int i = 0; i < BLOCK_K / 16; ++i) {
            int src_offset = i * 16;
            int dst_offset = i * 16 * BLOCK_M;
            AscendC::DataCopy(a1[dst_offset], a_gm[src_offset], { this->curr_block_m, 1, uint16_t(k / 16 - 1), 0});
        }
        // for b (transposed): dn -> zn
        for (int i = 0; i < BLOCK_K / 16; ++i) {
            int src_offset = i * 16;
            int dst_offset = i * 16 * BLOCK_N;
            AscendC::DataCopy(b1[dst_offset], b_gm[src_offset], { BLOCK_N, 1, uint16_t(k / 16 - 1), 0 });
        }

        a1_que.EnQue(a1);
        b1_que.EnQue(b1);
    }

    __aicore__ inline void CopyGmToL2Staged(int stage) {
        AscendC::LocalTensor<scalar_t> a1 = a1_que.AllocTensor<scalar_t>();
        AscendC::LocalTensor<scalar_t> b1 = b1_que.AllocTensor<scalar_t>();
        // for a: nd -> nz
        for (int i = 0; i < BLOCK_K / 16; ++i) {
            int src_offset = i * 16 + stage * BLOCK_K;
            int dst_offset = i * 16 * BLOCK_M;
            AscendC::DataCopy(a1[dst_offset], a_gm[src_offset], { this->curr_block_m, 1, uint16_t(k / 16 - 1), 0});
        }
        // for b (transposed): dn -> zn
        for (int i = 0; i < BLOCK_K / 16; ++i) {
            int src_offset = i * 16 + stage * BLOCK_K;
            int dst_offset = i * 16 * BLOCK_N;
            AscendC::DataCopy(b1[dst_offset], b_gm[src_offset], { BLOCK_N, 1, uint16_t(k / 16 - 1), 0 });
        }

        a1_que.EnQue(a1);
        b1_que.EnQue(b1);
    }

    __aicore__ inline void CopyL2ToL1() {
        // for a: nz -> zz
        AscendC::LocalTensor<scalar_t> a2 = a2_que.AllocTensor<scalar_t>();
        AscendC::LocalTensor<scalar_t> a1 = a1_que.DeQue<scalar_t>();
        for (int i = 0; i < BLOCK_M / 16; ++i) {
            int src_offset = i * 16 * 16;
            int dst_offset = i * 16 * BLOCK_K;
            AscendC::LoadData2dParams params;
            params.repeatTimes = BLOCK_K / 16;
            params.srcStride = BLOCK_M / 16;
            params.ifTranspose = false;
            AscendC::LoadData(a2[dst_offset], a1[src_offset], params);
        }
        a2_que.EnQue(a2);
        a1_que.FreeTensor(a1);

        // for b: zn -> zn
        AscendC::LocalTensor<scalar_t> b2 = b2_que.AllocTensor<scalar_t>();
        AscendC::LocalTensor<scalar_t> b1 = b1_que.DeQue<scalar_t>();
        AscendC::LoadData2dParams params;
        params.repeatTimes = (BLOCK_K * BLOCK_N) / (16 * 16);
        params.srcStride = 1;
        params.ifTranspose = false;
        AscendC::LoadData(b2, b1, params);

        b2_que.EnQue(b2);
        b1_que.FreeTensor(b1);
    }

    __aicore__ inline void Mma(AscendC::LocalTensor<acc_t>& acc, bool zeroed) {
        AscendC::LocalTensor<scalar_t> a2 = a2_que.DeQue<scalar_t>();
        AscendC::LocalTensor<scalar_t> b2 = b2_que.DeQue<scalar_t>();
        AscendC::MmadParams params;
        params.m = BLOCK_M;
        params.n = BLOCK_N;
        params.k = BLOCK_K;
        params.cmatrixInitVal = zeroed;
        AscendC::Mmad(acc, a2, b2, params);
        // for (int i = 0; i < 10; ++i) {
        //     AscendC::MmadParams params;
        //     params.m = BLOCK_M;
        //     params.n = BLOCK_N;
        //     params.k = BLOCK_K;
        //     params.cmatrixInitVal = i == 0;
        //     AscendC::Mmad(acc, a2, b2, params);
        // }

        a2_que.FreeTensor(a2);
        b2_que.FreeTensor(b2);
    }

    __aicore__ inline void CopyCO1ToCO2() {
        AscendC::LocalTensor<acc_t> co1 = co1_que.DeQue<acc_t>();
        AscendC::LocalTensor<acc_t> co2 = co2_que.AllocTensor<acc_t>();

        // for acc: nz -> nz
        AscendC::DataCopyParams params;
        params.blockCount = 1;
        params.blockLen = (BLOCK_M * BLOCK_N) / (16 * 16);
        AscendC::DataCopyEnhancedParams enhanced_params;
        enhanced_params.blockMode = AscendC::BlockMode::BLOCK_MODE_MATRIX;
        AscendC::DataCopy(co2, co1, params, enhanced_params);
        co2_que.EnQue(co2);
        co1_que.FreeTensor(co1);
    }

    __aicore__ inline void CopyCO2ToGm() {
        AscendC::LocalTensor<acc_t> co2 = co2_que.DeQue<acc_t>();
        AscendC::LocalTensor<scalar_t> c_cast = c_que.AllocTensor<scalar_t>();
        Cast(c_cast, co2, AscendC::RoundMode::CAST_ODD, BLOCK_M * BLOCK_N);
        c_que.EnQue(c_cast);
        co2_que.FreeTensor(co2);

        AscendC::LocalTensor<scalar_t> c_casted = c_que.DeQue<scalar_t>();
        // for acc: nz -> nd
        for (int i = 0; i < BLOCK_N / 16; ++i) {
            int src_offset = i * BLOCK_M * 16;
            int dst_offset = i * 16;
            // FIXME: uint16_t out of bound for large n
            // AscendC::DataCopy(c_gm[dst_offset], co2[src_offset], { this->curr_block_m, 2, 0, uint16_t((n / 16 - 1) * 2) });
            AscendC::DataCopy(c_gm[dst_offset], c_casted[src_offset], { this->curr_block_m, 1, 0, uint16_t(n / 16 - 1) });
        }
        c_que.FreeTensor(c_casted);
    }
};


#endif

