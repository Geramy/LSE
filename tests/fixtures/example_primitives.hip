// Several primitives sharing device helpers in one file.

__device__ inline float lse_softplus(float x) {
  return __logf(1.0f + __expf(x));
}

__device__ inline float lse_sigmoid_(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

// LSE_PRIMITIVE: example.mish arity=1
__device__ inline float lse_mish(float x) {
  return x * tanhf(lse_softplus(x));
}

// LSE_PRIMITIVE: example.swiglu arity=2
__device__ inline float lse_swiglu(float gate, float up) {
  return up * gate * lse_sigmoid_(gate);
}

// LSE_PRIMITIVE: example.clip3 arity=3
__device__ inline float lse_clip3(float x, float lo, float hi) {
  return fminf(fmaxf(x, lo), hi);
}
