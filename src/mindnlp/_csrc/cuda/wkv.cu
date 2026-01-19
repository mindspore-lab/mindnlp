#include <stdio.h>
#include <assert.h>

#define MIN_VALUE (-1e38)

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C,
                               const F *__restrict__ const _w, const F *__restrict__ const _u, const F *__restrict__ const _k, const F *__restrict__ const _v,
                               F *__restrict__ const _y) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;

    F u = _u[_c];
    F w = _w[_c];
    const F *__restrict__ const k = _k + _offset;
    const F *__restrict__ const v = _v + _offset;
    F *__restrict__ const y = _y + _offset;

    F p = 0, q = 0, o = MIN_VALUE;
    // p and q are running sums divided by exp(o) (to avoid overflows)
    for (int i = 0; i < T; i++) {
        const int ii = i * C;

        F no = max(o, u + k[ii]);
        F A = exp(o - no);
        F B = exp(u + k[ii] - no);
        y[ii] = (A * p + B * v[ii]) / (A * q + B);

        no = max(w + o, k[ii]);
        A = exp(w + o - no);
        B = exp(k[ii] - no);
        p = A * p + B * v[ii];
        q = A * q + B;
        o = no;
    }
}

template <typename F>
__global__ void kernel_forward_with_state(
    const int B, const int T, const int C, const F *__restrict__ const _w, const F *__restrict__ const _u,
    const F *__restrict__ const _k, const F *__restrict__ const _v, F *__restrict__ const _y, F *__restrict__ const _s
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset_s = _b * C * 3 + _c * 3;
    const int _offset = _b * T * C + _c;

    F u = _u[_c];
    F w = _w[_c];
    const F *__restrict__ const k = _k + _offset;
    const F *__restrict__ const v = _v + _offset;
    F *__restrict__ const y = _y + _offset;
    F *__restrict__ const s = _s + _offset_s;

    // aa and bb are running sums divided by exp(pp) (to avoid overflow)
    F aa = s[0], bb = s[1], pp = s[2];
    for (int i = 0; i < T; i++) {
        const int ii = i * C;
        const F kk = k[ii];
        const F vv = v[ii];

        F ww = u + kk;
        F p = max(pp, ww);
        F e1 = exp(pp - p);
        F e2 = exp(ww - p);
        y[ii] = (e1 * aa + e2 * vv) / (e1 * bb + e2);
        
        ww = w + pp;
        p = max(ww, kk);
        e1 = exp(ww - p);
        e2 = exp(kk - p);
        aa = e1 * aa + e2 * vv;
        bb = e1 * bb + e2;
        pp = p;
    }
    s[0] = aa;
    s[1] = bb;
    s[2] = pp;
}

template <typename F>
__global__ void kernel_backward(const int B, const int T, const int C,
                                const F *__restrict__ const _w, const F *__restrict__ const _u, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _gy,
                                F *__restrict__ const _gw, F *__restrict__ const _gu, F *__restrict__ const _gk, F *__restrict__ const _gv) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;

    F u = _u[_c];
    F w = _w[_c];
    const F *__restrict__ const k = _k + _offset;
    const F *__restrict__ const v = _v + _offset;
    const F *__restrict__ const gy = _gy + _offset;

    F *__restrict__ const gk = _gk + _offset;
    F *__restrict__ const gv = _gv + _offset;

    F y[Tmax], z[Tmax], zexp[Tmax];

    F gw = 0, gu = 0;
    F p = 0, q = 0;
    F dpdw = 0, dqdw = 0;
    F o = MIN_VALUE;
    for (int i = 0; i < T; i++) {
        const int ii = i * C;
        F no = max(o, k[ii] + u);
        F A = exp(o - no);
        F B = exp(k[ii] + u - no);

        F num = A * p + B * v[ii];
        F iden = 1 / (A * q + B);

        y[i] = num * iden;
        z[i] = iden;
        zexp[i] = k[ii] + u - no;

        gw += gy[ii] * (dpdw - dqdw * y[i]) * iden * A;
        gu += gy[ii] * (v[ii] - y[i]) * B * iden;

        no = max(w + o, k[ii]);
        A = exp(w + o - no);
        B = exp(k[ii] - no);
        dpdw = A * (p + dpdw);
        dqdw = A * (q + dqdw);
        p = A * p + B * v[ii];
        q = A * q + B;
        o = no;
    }

    F gp = 0, gq = 0;
    o = MIN_VALUE;
    for (int i = T - 1; i >= 0; i--) {
        const int ii = i * C;
        F A = gy[ii] * z[i] * exp(zexp[i]);
        F B = exp(k[ii] + o);
        gk[ii] = A * (v[ii] - y[i]) + B * (gp * v[ii] + gq);
        gv[ii] = A + B * gp;

        F no = max(w + o, zexp[i] - k[ii] - u);
        A = exp(w + o - no);
        B = gy[ii] * z[i] * exp(zexp[i] - k[ii] - u - no);
        gp = A * gp + B;
        gq = A * gq - B * y[i];
        o = no;
    }

    // Multiply by w because the w -> -exp(w) preprocessing is halfway in the backwards pass, even though it's not in the forward pass
    const int _offsetBC = _b * C + _c;
    _gw[_offsetBC] += gw * _w[_c];
    _gu[_offsetBC] += gu;
}

extern "C" {

int wkv_forward(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                void *extra) {
    cudaStream_t custream = static_cast<cudaStream_t>(stream);
    if (nparam != 5) return 1;
    float *w = static_cast<float *>(params[0]);
    float *u = static_cast<float *>(params[1]);
    float *k = static_cast<float *>(params[2]);
    float *v = static_cast<float *>(params[3]);
    float *y = static_cast<float *>(params[4]);

    int B = static_cast<int>(shapes[2][0]);
    int T = static_cast<int>(shapes[2][1]);
    int C = static_cast<int>(shapes[2][2]);

    dim3 threadsPerBlock( min(C, 32) ); // requires --maxrregcount 60 for optimal performance
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_forward<<<numBlocks, threadsPerBlock, 0, custream>>>(B, T, C, w, u, k, v, y);
    return 0;
}

int wkv_forward_with_state(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                           void *extra) {
    cudaStream_t custream = static_cast<cudaStream_t>(stream);
    if (nparam != 6) return 1;
    float *w = static_cast<float *>(params[0]);
    float *u = static_cast<float *>(params[1]);
    float *k = static_cast<float *>(params[2]);
    float *v = static_cast<float *>(params[3]);
    float *s = static_cast<float *>(params[4]);
    float *y = static_cast<float *>(params[5]);

    int B = static_cast<int>(shapes[2][0]);
    int T = static_cast<int>(shapes[2][1]);
    int C = static_cast<int>(shapes[2][2]);

    dim3 threadsPerBlock( min(C, 32) ); // requires --maxrregcount 60 for optimal performance
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_forward_with_state<<<numBlocks, threadsPerBlock, 0, custream>>>(B, T, C, w, u, k, v, y, s);
    return 0;
}

int wkv_backward(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                 void *extra) {
    cudaStream_t custream = static_cast<cudaStream_t>(stream);
    if (nparam != 9) return 1;
    float *w = static_cast<float *>(params[0]);
    float *u = static_cast<float *>(params[1]);
    float *k = static_cast<float *>(params[2]);
    float *v = static_cast<float *>(params[3]);
    float *gy = static_cast<float *>(params[4]);
    float *gw = static_cast<float *>(params[5]);
    float *gu = static_cast<float *>(params[6]);
    float *gk = static_cast<float *>(params[7]);
    float *gv = static_cast<float *>(params[8]);

    int B = static_cast<int>(shapes[2][0]);
    int T = static_cast<int>(shapes[2][1]);
    int C = static_cast<int>(shapes[2][2]);

    dim3 threadsPerBlock( min(C, 32) ); // requires --maxrregcount 60 for optimal performance
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_backward<<<numBlocks, threadsPerBlock, 0, custream>>>(B, T, C, w, u, k, v, gy, gw, gu, gk, gv);
    return 0;
}

}