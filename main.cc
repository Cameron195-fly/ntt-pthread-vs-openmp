#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <omp.h>
#include <algorithm>
#include <pthread.h>
#include <cstdlib>

using ll = long long;
using i128 = __int128_t;

#define MAX_THREADS 8
#define CACHE_LINE 64

const ll CRT_MODS[3] = {7340033LL, 104857601LL, 469762049LL};
const i128 CRT_PRODUCT = (i128)CRT_MODS[0] * CRT_MODS[1] * CRT_MODS[2];

// ============================ 文件IO ============================

void fRead(ll *a, ll *b, int *n, ll *p, int input_id) {
    std::string str1 = "/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strin = str1 + str2 + ".in";
    char data_path[strin.size() + 1];
    std::copy(strin.begin(), strin.end(), data_path);
    data_path[strin.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    fin >> *n >> *p;
    for (int i = 0; i < *n; i++) fin >> a[i];
    for (int i = 0; i < *n; i++) fin >> b[i];
}

void fCheck(ll *ab, int n, int input_id) {
    std::string str1 = "/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char data_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), data_path);
    data_path[strout.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    for (int i = 0; i < n * 2 - 1; i++) {
        long long x;
        fin >> x;
        if (x != ab[i]) {
            std::cout << "多项式乘法结果错误 (index=" << i
                      << " expected=" << x << " got=" << ab[i] << ")" << std::endl;
            return;
        }
    }
    std::cout << "多项式乘法结果正确" << std::endl;
}

void fWrite(ll *ab, int n, int input_id) {
    std::string str1 = "files/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char output_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), output_path);
    output_path[strout.size()] = '\0';
    std::ofstream fout;
    fout.open(output_path, std::ios::out);
    for (int i = 0; i < n * 2 - 1; i++) fout << ab[i] << '\n';
}

// ============================ 基础工具 ============================

ll qpow(ll a, ll b, ll mod) {
    ll res = 1; a %= mod;
    while (b) {
        if (b & 1) res = (i128)res * a % mod;
        a = (i128)a * a % mod; b >>= 1;
    }
    return res;
}

ll exgcd(ll a, ll b, ll &x, ll &y) {
    if (b == 0) { x = 1; y = 0; return a; }
    ll d = exgcd(b, a % b, y, x);
    y -= (a / b) * x; return d;
}

ll mod_inv(ll a, ll mod) {
    ll x, y; exgcd(a, mod, x, y);
    return (x % mod + mod) % mod;
}

// ============================ 位逆序 + 串行NTT ============================

static ll tmp_a[600000], tmp_b[600000];

void bit_reverse(ll *a, int n) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
}

void ntt_serial(ll *a, int n, ll p, int inv) {
    bit_reverse(a, n);
    for (int len = 2; len <= n; len <<= 1) {
        ll wlen = qpow(3, (p - 1) / len, p);
        if (inv == -1) wlen = qpow(wlen, p - 2, p);
        for (int i = 0; i < n; i += len) {
            ll w = 1;
            for (int j = 0; j < len / 2; j++) {
                ll u = a[i + j];
                ll v = (i128)a[i + j + len / 2] * w % p;
                a[i + j] = (u + v < p) ? u + v : u + v - p;
                a[i + j + len / 2] = (u >= v) ? u - v : u - v + p;
                w = (i128)w * wlen % p;
            }
        }
    }
    if (inv == -1) {
        ll n_inv = qpow(n, p - 2, p);
        for (int i = 0; i < n; i++) a[i] = (i128)a[i] * n_inv % p;
    }
}

// ============================ Pthread NTT（水平划分） ============================

static ll w_buf_pt[MAX_THREADS][300000];
pthread_barrier_t pt_barrier;

struct PthreadParam {
    ll *a; int n; ll p; int inv; int tid; int nthreads;
};

void *pthread_ntt_worker(void *arg) {
    PthreadParam *tp = (PthreadParam *)arg;
    ll *a = tp->a; int n = tp->n; ll mod = tp->p;
    int inv = tp->inv; int tid = tp->tid; int nt = tp->nthreads;
    ll *wbuf = w_buf_pt[tid];

    for (int len = 2; len <= n; len <<= 1) {
        ll wlen = qpow(3, (mod - 1) / len, mod);
        if (inv == -1) wlen = qpow(wlen, mod - 2, mod);
        wbuf[0] = 1;
        for (int j = 1; j < len / 2; j++)
            wbuf[j] = (i128)wbuf[j - 1] * wlen % mod;
        pthread_barrier_wait(&pt_barrier);
        int total_blocks = n / len;
        int base = total_blocks / nt, rem = total_blocks % nt;
        int sb, eb;
        if (tid < rem) { sb = tid * (base + 1); eb = sb + base + 1; }
        else { sb = rem * (base + 1) + (tid - rem) * base; eb = sb + base; }
        for (int b = sb; b < eb; b++) {
            int i = b * len;
            for (int j = 0; j < len / 2; j++) {
                ll u = a[i + j];
                ll v = (i128)a[i + j + len / 2] * wbuf[j] % mod;
                a[i + j] = (u + v < mod) ? u + v : u + v - mod;
                a[i + j + len / 2] = (u >= v) ? u - v : u - v + mod;
            }
        }
        pthread_barrier_wait(&pt_barrier);
    }
    if (inv == -1) {
        pthread_barrier_wait(&pt_barrier);
        ll n_inv = qpow(n, mod - 2, mod);
        int base = n / nt, rem = n % nt;
        int s, e;
        if (tid < rem) { s = tid * (base + 1); e = s + base + 1; }
        else { s = rem * (base + 1) + (tid - rem) * base; e = s + base; }
        for (int i = s; i < e; i++) a[i] = (i128)a[i] * n_inv % mod;
    }
    pthread_exit(NULL);
}

void ntt_pthread(ll *a, int n, ll p, int inv, int nt) {
    if (nt == 1) { ntt_serial(a, n, p, inv); return; }
    bit_reverse(a, n);
    pthread_barrier_init(&pt_barrier, NULL, nt);
    PthreadParam params[MAX_THREADS];
    pthread_t handles[MAX_THREADS];
    for (int t = 0; t < nt; t++) {
        params[t] = {a, n, p, inv, t, nt};
        pthread_create(&handles[t], NULL, pthread_ntt_worker, &params[t]);
    }
    for (int t = 0; t < nt; t++) pthread_join(handles[t], NULL);
    pthread_barrier_destroy(&pt_barrier);
}

// ============================ OpenMP NTT（水平划分） ============================

static ll w_buf_omp[300000];

void ntt_openmp(ll *a, int n, ll p, int inv, int nt) {
    omp_set_num_threads(nt);
    bit_reverse(a, n);
    for (int len = 2; len <= n; len <<= 1) {
        ll wlen = qpow(3, (p - 1) / len, p);
        if (inv == -1) wlen = qpow(wlen, p - 2, p);
        w_buf_omp[0] = 1;
        for (int j = 1; j < len / 2; j++)
            w_buf_omp[j] = (i128)w_buf_omp[j - 1] * wlen % p;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < len / 2; j++) {
                ll u = a[i + j];
                ll v = (i128)a[i + j + len / 2] * w_buf_omp[j] % p;
                a[i + j] = (u + v < p) ? u + v : u + v - p;
                a[i + j + len / 2] = (u >= v) ? u - v : u - v + p;
            }
        }
    }
    if (inv == -1) {
        ll n_inv = qpow(n, p - 2, p);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) a[i] = (i128)a[i] * n_inv % p;
    }
}

// ============================ OpenMP NTT（垂直划分 - 对比） ============================

void ntt_omp_vertical(ll *a, int n, ll p, int inv, int nt) {
    omp_set_num_threads(nt);
    bit_reverse(a, n);
    for (int len = 2; len <= n; len <<= 1) {
        ll wlen = qpow(3, (p - 1) / len, p);
        if (inv == -1) wlen = qpow(wlen, p - 2, p);
        w_buf_omp[0] = 1;
        for (int j = 1; j < len / 2; j++)
            w_buf_omp[j] = (i128)w_buf_omp[j - 1] * wlen % p;
        for (int i = 0; i < n; i += len) {
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < len / 2; j++) {
                ll u = a[i + j];
                ll v = (i128)a[i + j + len / 2] * w_buf_omp[j] % p;
                a[i + j] = (u + v < p) ? u + v : u + v - p;
                a[i + j + len / 2] = (u >= v) ? u - v : u - v + p;
            }
        }
    }
    if (inv == -1) {
        ll n_inv = qpow(n, p - 2, p);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) a[i] = (i128)a[i] * n_inv % p;
    }
}

// ============================ OpenMP NTT（循环展开） ============================

void ntt_omp_unroll(ll *a, int n, ll p, int inv, int nt) {
    omp_set_num_threads(nt);
    bit_reverse(a, n);
    for (int len = 2; len <= n; len <<= 1) {
        ll wlen = qpow(3, (p - 1) / len, p);
        if (inv == -1) wlen = qpow(wlen, p - 2, p);
        w_buf_omp[0] = 1;
        for (int j = 1; j < len / 2; j++)
            w_buf_omp[j] = (i128)w_buf_omp[j - 1] * wlen % p;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i += len) {
            int j = 0;
            for (; j + 3 < len / 2; j += 4) {
                ll u0 = a[i+j],     v0 = (i128)a[i+j+len/2]     * w_buf_omp[j]   % p;
                ll u1 = a[i+j+1],   v1 = (i128)a[i+j+1+len/2]   * w_buf_omp[j+1] % p;
                ll u2 = a[i+j+2],   v2 = (i128)a[i+j+2+len/2]   * w_buf_omp[j+2] % p;
                ll u3 = a[i+j+3],   v3 = (i128)a[i+j+3+len/2]   * w_buf_omp[j+3] % p;
                a[i+j]         = (u0+v0<p)?u0+v0:u0+v0-p;
                a[i+j+1]       = (u1+v1<p)?u1+v1:u1+v1-p;
                a[i+j+2]       = (u2+v2<p)?u2+v2:u2+v2-p;
                a[i+j+3]       = (u3+v3<p)?u3+v3:u3+v3-p;
                a[i+j+len/2]   = (u0>=v0)?u0-v0:u0-v0+p;
                a[i+j+1+len/2] = (u1>=v1)?u1-v1:u1-v1+p;
                a[i+j+2+len/2] = (u2>=v2)?u2-v2:u2-v2+p;
                a[i+j+3+len/2] = (u3>=v3)?u3-v3:u3-v3+p;
            }
            for (; j < len / 2; j++) {
                ll u = a[i + j];
                ll v = (i128)a[i + j + len / 2] * w_buf_omp[j] % p;
                a[i + j] = (u + v < p) ? u + v : u + v - p;
                a[i + j + len / 2] = (u >= v) ? u - v : u - v + p;
            }
        }
    }
    if (inv == -1) {
        ll n_inv = qpow(n, p - 2, p);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) a[i] = (i128)a[i] * n_inv % p;
    }
}

// ============================ CRT ============================

struct CRTContext {
    ll mods[3];
    ll inv_m0_mod_m1, inv_m01_mod_m2;
    bool initialized;
    CRTContext() : initialized(false) {}
    void init(ll m0, ll m1, ll m2) {
        mods[0] = m0; mods[1] = m1; mods[2] = m2;
        inv_m0_mod_m1 = mod_inv(mods[0] % mods[1], mods[1]);
        inv_m01_mod_m2 = mod_inv((mods[0] * mods[1]) % mods[2], mods[2]);
        initialized = true;
    }
    ll garner(ll r0, ll r1, ll r2, ll target) const {
        ll diff1 = ((r1 - r0) % mods[1] + mods[1]) % mods[1];
        ll t1 = (i128)diff1 * inv_m0_mod_m1 % mods[1];
        i128 x1 = (i128)r0 + (i128)mods[0] * t1;
        ll diff2 = ((r2 - (ll)(x1 % mods[2])) % mods[2] + mods[2]) % mods[2];
        ll t2 = (i128)diff2 * inv_m01_mod_m2 % mods[2];
        i128 result = x1 + (i128)(mods[0] * mods[1]) * t2;
        ll ans = (ll)(result % target);
        return (ans < 0) ? ans + target : ans;
    }
};

static CRTContext crt_ctx;
static ll crt_buf[3][600000], crt_tmp_a[3][600000], crt_tmp_b[3][600000];

struct CRTParam { int tid; ll *a, *b, *res; int n; ll mod; };

void *crt_worker(void *arg) {
    CRTParam *tp = (CRTParam *)arg;
    int tid = tp->tid, n = tp->n;
    ll mod = tp->mod;
    int len = 1; while (len < n * 2) len <<= 1;
    ll *la = crt_tmp_a[tid], *lb = crt_tmp_b[tid];
    for (int i = 0; i < len; i++) {
        la[i] = (i < n) ? ((tp->a[i] % mod) + mod) % mod : 0;
        lb[i] = (i < n) ? ((tp->b[i] % mod) + mod) % mod : 0;
    }
    ntt_serial(la, len, mod, 1);
    ntt_serial(lb, len, mod, 1);
    for (int i = 0; i < len; i++) la[i] = (i128)la[i] * lb[i] % mod;
    ntt_serial(la, len, mod, -1);
    for (int i = 0; i < n * 2 - 1; i++) tp->res[i] = la[i];
    pthread_exit(NULL);
}

bool crt_applicable(int n, ll p) {
    i128 max_coeff = (i128)(n - 1) * (p - 1) * (p - 1);
    return max_coeff < CRT_PRODUCT / 2;
}

void poly_crt(ll *a, ll *b, ll *ab, int n, ll P) {
    if (!crt_ctx.initialized) crt_ctx.init(CRT_MODS[0], CRT_MODS[1], CRT_MODS[2]);
    CRTParam params[3]; pthread_t handles[3];
    for (int t = 0; t < 3; t++) {
        params[t] = {t, a, b, crt_buf[t], n, CRT_MODS[t]};
        pthread_create(&handles[t], NULL, crt_worker, &params[t]);
    }
    for (int t = 0; t < 3; t++) pthread_join(handles[t], NULL);
    int m = n * 2 - 1;
    for (int i = 0; i < m; i++)
        ab[i] = crt_ctx.garner(crt_buf[0][i], crt_buf[1][i], crt_buf[2][i], P);
}

// ============================ 多项式乘法包装器 ============================

static ll buf_a[600000], buf_b[600000];

void poly_serial(ll *a, ll *b, ll *ab, int n, ll p) {
    int len = 1; while (len < n * 2) len <<= 1;
    for (int i = 0; i < len; i++) {
        tmp_a[i] = (i < n) ? a[i] : 0;
        tmp_b[i] = (i < n) ? b[i] : 0;
    }
    ntt_serial(tmp_a, len, p, 1); ntt_serial(tmp_b, len, p, 1);
    for (int i = 0; i < len; i++) tmp_a[i] = (i128)tmp_a[i] * tmp_b[i] % p;
    ntt_serial(tmp_a, len, p, -1);
    for (int i = 0; i < n * 2 - 1; i++) ab[i] = tmp_a[i];
}

void poly_pthread(ll *a, ll *b, ll *ab, int n, ll p, int nt) {
    int len = 1; while (len < n * 2) len <<= 1;
    for (int i = 0; i < len; i++) {
        buf_a[i] = (i < n) ? a[i] : 0;
        buf_b[i] = (i < n) ? b[i] : 0;
    }
    ntt_pthread(buf_a, len, p, 1, nt); ntt_pthread(buf_b, len, p, 1, nt);
    for (int i = 0; i < len; i++) buf_a[i] = (i128)buf_a[i] * buf_b[i] % p;
    ntt_pthread(buf_a, len, p, -1, nt);
    for (int i = 0; i < n * 2 - 1; i++) ab[i] = buf_a[i];
}

void poly_openmp(ll *a, ll *b, ll *ab, int n, ll p, int nt) {
    int len = 1; while (len < n * 2) len <<= 1;
    for (int i = 0; i < len; i++) {
        buf_a[i] = (i < n) ? a[i] : 0;
        buf_b[i] = (i < n) ? b[i] : 0;
    }
    ntt_openmp(buf_a, len, p, 1, nt); ntt_openmp(buf_b, len, p, 1, nt);
    #pragma omp parallel for if(nt > 1)
    for (int i = 0; i < len; i++) buf_a[i] = (i128)buf_a[i] * buf_b[i] % p;
    ntt_openmp(buf_a, len, p, -1, nt);
    for (int i = 0; i < n * 2 - 1; i++) ab[i] = buf_a[i];
}

void poly_openmp_vertical(ll *a, ll *b, ll *ab, int n, ll p, int nt) {
    int len = 1; while (len < n * 2) len <<= 1;
    for (int i = 0; i < len; i++) {
        buf_a[i] = (i < n) ? a[i] : 0;
        buf_b[i] = (i < n) ? b[i] : 0;
    }
    ntt_omp_vertical(buf_a, len, p, 1, nt);
    ntt_omp_vertical(buf_b, len, p, 1, nt);
    #pragma omp parallel for if(nt > 1)
    for (int i = 0; i < len; i++) buf_a[i] = (i128)buf_a[i] * buf_b[i] % p;
    ntt_omp_vertical(buf_a, len, p, -1, nt);
    for (int i = 0; i < n * 2 - 1; i++) ab[i] = buf_a[i];
}

void poly_openmp_unroll(ll *a, ll *b, ll *ab, int n, ll p, int nt) {
    int len = 1; while (len < n * 2) len <<= 1;
    for (int i = 0; i < len; i++) {
        buf_a[i] = (i < n) ? a[i] : 0;
        buf_b[i] = (i < n) ? b[i] : 0;
    }
    ntt_omp_unroll(buf_a, len, p, 1, nt);
    ntt_omp_unroll(buf_b, len, p, 1, nt);
    #pragma omp parallel for if(nt > 1)
    for (int i = 0; i < len; i++) buf_a[i] = (i128)buf_a[i] * buf_b[i] % p;
    ntt_omp_unroll(buf_a, len, p, -1, nt);
    for (int i = 0; i < n * 2 - 1; i++) ab[i] = buf_a[i];
}

// ============================ 测试函数 ============================

void self_test(int n, ll p, int version, int num_threads) {
    static ll ta[300000], tb[300000], tab[300000];
    for (int i = 0; i < n; i++) { ta[i] = rand() % p; tb[i] = rand() % p; }
    memset(tab, 0, sizeof(tab));
    auto s = std::chrono::high_resolution_clock::now();
    switch (version) {
        case 0: poly_serial(ta, tb, tab, n, p); break;
        case 1: poly_pthread(ta, tb, tab, n, p, num_threads); break;
        case 2: poly_openmp(ta, tb, tab, n, p, num_threads); break;
        case 3: poly_openmp_vertical(ta, tb, tab, n, p, num_threads); break;
        case 4: poly_openmp_unroll(ta, tb, tab, n, p, num_threads); break;
        case 5: poly_crt(ta, tb, tab, n, p); break;
        default: poly_serial(ta, tb, tab, n, p); break;
    }
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::ratio<1, 1000>> d = e - s;
    const char *names[] = {"Serial","Pthread","OpenMP_h","OpenMP_v","OpenMP_unroll","CRT"};
    std::cout << "[self_test] n=" << n << " " << names[version]
              << " " << d.count() << " ms" << std::endl;
}

void align_test(int n, ll p, int nt) {
    int len = 1; while (len < n * 2) len <<= 1;
    ll *a1 = (ll *)aligned_alloc(64, len * sizeof(ll));
    ll *a2_raw = (ll *)malloc(len * sizeof(ll) + 64);
    ll *a2 = (ll *)(((uintptr_t)a2_raw + 63) & ~63) + 1;
    for (int i = 0; i < len; i++) { a1[i] = rand() % p; a2[i] = a1[i]; }
    ntt_pthread(a1, len, p, 1, nt); ntt_pthread(a2, len, p, 1, nt);
    auto s1 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < 10; t++) ntt_pthread(a1, len, p, 1, nt);
    auto e1 = std::chrono::high_resolution_clock::now();
    auto s2 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < 10; t++) ntt_pthread(a2, len, p, 1, nt);
    auto e2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::ratio<1, 1000>> d1 = e1 - s1, d2 = e2 - s2;
    std::cout << "[align_test] aligned=" << (d1.count()/10.0)
              << " ms unaligned=" << (d2.count()/10.0)
              << " ms ratio=" << (d2.count()/d1.count()) << std::endl;
    free(a1); free(a2_raw);
}

void horizontal_vs_vertical_test(int n, ll p, int nt) {
    static ll ta[300000], tb[300000], tab[300000];
    for (int i = 0; i < n; i++) { ta[i] = rand() % p; tb[i] = rand() % p; }
    std::cout << "\n--- Horizontal vs Vertical n=" << n << " t=" << nt << " ---" << std::endl;
    memset(tab, 0, sizeof(tab));
    auto s1 = std::chrono::high_resolution_clock::now();
    poly_openmp(ta, tb, tab, n, p, nt);
    auto e1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::ratio<1, 1000>> d1 = e1 - s1;
    memset(tab, 0, sizeof(tab));
    auto s2 = std::chrono::high_resolution_clock::now();
    poly_openmp_vertical(ta, tb, tab, n, p, nt);
    auto e2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::ratio<1, 1000>> d2 = e2 - s2;
    std::cout << "  Horizontal: " << d1.count() << " ms" << std::endl;
    std::cout << "  Vertical:   " << d2.count() << " ms" << std::endl;
    std::cout << "  Ratio(v/h): " << std::fixed << std::setprecision(2) << (d2.count()/d1.count()) << std::endl;
}

void unroll_test(int n, ll p, int nt) {
    static ll ta[300000], tb[300000], tab[300000];
    for (int i = 0; i < n; i++) { ta[i] = rand() % p; tb[i] = rand() % p; }
    std::cout << "\n--- Loop Unrolling n=" << n << " t=" << nt << " ---" << std::endl;
    memset(tab, 0, sizeof(tab));
    auto s1 = std::chrono::high_resolution_clock::now();
    poly_openmp(ta, tb, tab, n, p, nt);
    auto e1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::ratio<1, 1000>> d1 = e1 - s1;
    memset(tab, 0, sizeof(tab));
    auto s2 = std::chrono::high_resolution_clock::now();
    poly_openmp_unroll(ta, tb, tab, n, p, nt);
    auto e2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::ratio<1, 1000>> d2 = e2 - s2;
    std::cout << "  Standard:  " << d1.count() << " ms" << std::endl;
    std::cout << "  Unroll(x4):" << d2.count() << " ms" << std::endl;
    std::cout << "  Speedup:   " << std::fixed << std::setprecision(2) << (d1.count()/d2.count()) << std::endl;
}

void thread_overhead_test(int n, ll p) {
    std::cout << "\n--- Thread Overhead n=" << n << " ---" << std::endl;
    int scales_t[] = {1, 2, 4, 8};
    for (int nt : scales_t) {
        static ll ta[300000], tb[300000], tab[300000];
        for (int i = 0; i < n; i++) { ta[i] = rand() % p; tb[i] = rand() % p; }
        auto s = std::chrono::high_resolution_clock::now();
        poly_pthread(ta, tb, tab, n, p, nt);
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::ratio<1, 1000>> d = e - s;
        std::cout << "  threads=" << nt << ": " << d.count() << " ms" << std::endl;
    }
}

// ============================ 性能测试框架 ============================

ll global_a[300000], global_b[300000], global_ab[300000];

template<typename Func>
double mtimer(Func &&f, int w, int r) {
    for (int i = 0; i < w; i++) f();
    auto s = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < r; i++) f();
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::ratio<1, 1000>> d = e - s;
    return d.count() / r;
}

bool same(ll *a, ll *b, int n) {
    for (int i = 0; i < n * 2 - 1; i++) if (a[i] != b[i]) return false;
    return true;
}

void bench(int n, ll p, int maxt) {
    static ll ta[300000], tb[300000], ref[300000], test[300000];
    for (int i = 0; i < n; i++) { ta[i] = rand() % p; tb[i] = rand() % p; }
    std::cout << "\n======== n=" << n << " p=" << p << " ========" << std::endl;
    memset(ref, 0, sizeof(ref));
    double ts = mtimer([&](){ memset(test, 0, sizeof(test)); poly_serial(ta, tb, test, n, p); }, 1, 5);
    memcpy(ref, test, sizeof(test));
    std::cout << "[Serial] " << ts << " ms" << std::endl;
    std::cout << "--- Pthread ---" << std::endl;
    for (int t = 1; t <= maxt; t *= 2) {
        memset(test, 0, sizeof(test));
        double tp = mtimer([&](){ memset(test, 0, sizeof(test)); poly_pthread(ta, tb, test, n, p, t); }, 1, 5);
        std::cout << "  t=" << t << " " << tp << " ms sp=" << std::fixed << std::setprecision(2) << (ts/tp)
                  << " " << (same(test, ref, n) ? "OK" : "FAIL") << std::endl;
    }
    std::cout << "--- OpenMP ---" << std::endl;
    for (int t = 1; t <= maxt; t *= 2) {
        memset(test, 0, sizeof(test));
        double tp = mtimer([&](){ memset(test, 0, sizeof(test)); poly_openmp(ta, tb, test, n, p, t); }, 1, 5);
        std::cout << "  t=" << t << " " << tp << " ms sp=" << std::fixed << std::setprecision(2) << (ts/tp)
                  << " " << (same(test, ref, n) ? "OK" : "FAIL") << std::endl;
    }
    if (crt_applicable(n, p)) {
        std::cout << "--- CRT ---" << std::endl;
        memset(test, 0, sizeof(test));
        double tc = mtimer([&](){ memset(test, 0, sizeof(test)); poly_crt(ta, tb, test, n, p); }, 1, 5);
        std::cout << "  CRT " << tc << " ms sp=" << std::fixed << std::setprecision(2) << (ts/tc)
                  << " " << (same(test, ref, n) ? "OK" : "FAIL") << std::endl;
    } else {
        std::cout << "--- CRT skipped ---" << std::endl;
    }
}

// ============================ perf专用纯性能测试 ============================

static void perf_bench(const char* mode, int n, ll p, int nt) {
    const int RUNS = 10;
    double times[RUNS];
    for (int r = 0; r < RUNS; ++r) {
        ll *A = (ll*)aligned_alloc(64, 2 * n * sizeof(ll));
        ll *B = (ll*)aligned_alloc(64, 2 * n * sizeof(ll));
        ll *AB = (ll*)aligned_alloc(64, 2 * n * sizeof(ll));
        for (int i = 0; i < n; ++i) { A[i] = i % p; B[i] = (i * 3 + 7) % p; }
        auto s = std::chrono::high_resolution_clock::now();
        if (strcmp(mode, "serial") == 0) {
            poly_serial(A, B, AB, n, p);
        } else if (strcmp(mode, "pthread") == 0) {
            poly_pthread(A, B, AB, n, p, nt);
        } else if (strcmp(mode, "openmp") == 0) {
            poly_openmp(A, B, AB, n, p, nt);
        } else if (strcmp(mode, "crt") == 0) {
            poly_crt(A, B, AB, n, p);
        }
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::ratio<1, 1000>> d = e - s;
        times[r] = d.count();
        free(A); free(B); free(AB);
    }
    std::sort(times, times + RUNS);
    double median = times[RUNS / 2];
    double sum = 0;
    for (int i = 1; i < RUNS - 1; ++i) sum += times[i];
    double avg = sum / (RUNS - 2);
    std::cout << "[" << mode << " n=" << n << " p=" << p << " T=" << nt << "] "
              << "median=" << median << " ms  avg(trimmed)=" << avg << " ms" << std::endl;
}

// ============================ main ============================

int main(int argc, char *argv[]) {
    // 保证输入的所有模数的原根均为 3, 且模数都能表示为 a \times 4 ^ k + 1 的形式
    // 输入模数分别为 7340033 104857601 469762049 263882790666241
    // 第四个模数超过了整型表示范围, 如果实现此模数意义下的多项式乘法需要修改框架
    // 对第四个模数的输入数据不做必要要求, 如果要自行探索大模数 NTT, 请在完成前三个模数的基础代码及优化后实现大模数 NTT
    // 输入文件共五个, 第一个输入文件 n = 4, 其余四个文件分别对应四个模数, n = 131072
    // 在实现快速数论变化前, 后四个测试样例运行时间较久, 推荐调试正确性时只使用输入文件 1

    // 解析参数：支持 --mode serial|pthread|openmp|crt [threads]
    const char* mode = nullptr;
    int max_threads = 8;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = argv[++i];
        } else if (argv[i][0] != '-') {
            max_threads = atoi(argv[i]);
        }
    }
    if (max_threads < 1 || max_threads > MAX_THREADS) max_threads = 8;

    crt_ctx.init(CRT_MODS[0], CRT_MODS[1], CRT_MODS[2]);

    // ========== perf纯净模式：只跑指定算法的性能测试 ==========
    if (mode) {
        int nt = max_threads;
        std::cout << "=== PERF MODE: " << mode << "  threads=" << nt << " ===" << std::endl;
        // 预热
        perf_bench(mode, 4096, 7340033, nt);
        std::cout << "--- warmup done ---" << std::endl;
        // 正式数据（3个不同规模）
        perf_bench(mode, 65536, 7340033, nt);
        perf_bench(mode, 131072, 7340033, nt);
        perf_bench(mode, 262144, 7340033, nt);
        return 0;
    }

    // ========== 完整测试模式（原有行为） ==========
    std::cout << "=== NTT 多线程实验  threads=" << max_threads << " ===" << std::endl;

    // 正确性测试
    std::cout << "\n==== 正确性测试 ====" << std::endl;
    for (int i = 0; i <= 4; ++i) {
        int n_; ll p_;
        fRead(global_a, global_b, &n_, &p_, i);

        memset(global_ab, 0, sizeof(global_ab));
        auto s0 = std::chrono::high_resolution_clock::now();
        poly_serial(global_a, global_b, global_ab, n_, p_);
        auto e0 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::ratio<1, 1000>> d0 = e0 - s0;
        std::cout << "[Serial  id=" << i << "] "; fCheck(global_ab, n_, i);
        std::cout << "  " << d0.count() << " ms" << std::endl;

        memset(global_ab, 0, sizeof(global_ab));
        auto s1 = std::chrono::high_resolution_clock::now();
        poly_pthread(global_a, global_b, global_ab, n_, p_, 4);
        auto e1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::ratio<1, 1000>> d1 = e1 - s1;
        std::cout << "[Pthread id=" << i << "] "; fCheck(global_ab, n_, i);
        std::cout << "  " << d1.count() << " ms" << std::endl;

        memset(global_ab, 0, sizeof(global_ab));
        auto s2 = std::chrono::high_resolution_clock::now();
        poly_openmp(global_a, global_b, global_ab, n_, p_, 4);
        auto e2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::ratio<1, 1000>> d2 = e2 - s2;
        std::cout << "[OpenMP  id=" << i << "] "; fCheck(global_ab, n_, i);
        std::cout << "  " << d2.count() << " ms" << std::endl;

        if (crt_applicable(n_, p_)) {
            memset(global_ab, 0, sizeof(global_ab));
            auto s3 = std::chrono::high_resolution_clock::now();
            poly_crt(global_a, global_b, global_ab, n_, p_);
            auto e3 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::ratio<1, 1000>> d3 = e3 - s3;
            std::cout << "[CRT     id=" << i << "] "; fCheck(global_ab, n_, i);
            std::cout << "  " << d3.count() << " ms" << std::endl;
        } else {
            std::cout << "[CRT     id=" << i << "] SKIPPED" << std::endl;
        }
        std::cout << std::endl;
        fWrite(global_ab, n_, i);
    }

    // 性能基准测试
    std::cout << "\n\n==== 性能基准测试 ====" << std::endl;
    bench(131072, 7340033, max_threads);
    bench(65536, 104857601, max_threads);
    bench(131072, 469762049, max_threads);
    std::cout << "\n==== CRT大模数测试 ====" << std::endl;
    bench(65536, 263882790666241LL, max_threads);

    // 规模扫描
    std::cout << "\n\n==== 规模扫描 ====" << std::endl;
    int scales[] = {1024, 4096, 16384, 65536, 131072};
    for (int sz : scales) {
        std::cout << "\n--- n=" << sz << " ---" << std::endl;
        self_test(sz, 7340033, 0, max_threads);
        self_test(sz, 7340033, 1, max_threads);
        self_test(sz, 7340033, 2, max_threads);
        if (crt_applicable(sz, 7340033)) self_test(sz, 7340033, 5, max_threads);
    }

    // 探索性测试
    std::cout << "\n\n==== 探索性测试 ====" << std::endl;
    horizontal_vs_vertical_test(131072, 7340033, 4);
    horizontal_vs_vertical_test(131072, 7340033, 8);
    unroll_test(131072, 7340033, 4);
    align_test(131072, 7340033, 4);
    thread_overhead_test(131072, 7340033);

    return 0;
}