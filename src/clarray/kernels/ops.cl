#define MAX_NDIM 10

inline void ws_to_index(
  const uint i,
  __constant const uint* base_strides,
  const uint ndim,
  __private uint* index
) {
  uint j = i;
  for (uint d = 0; d < ndim; d++) {
    index[d] = (uint) (j / base_strides[d]);
    j = i % base_strides[d];
  }
}

__kernel void ew_add_fwd(
    __global const scalar_t* lhs,
    __constant const uint* lhs_strides,
    __global const scalar_t* rhs,
    __constant const uint* rhs_strides,
    __global scalar_t* output,
    __constant const uint* base_strides,
    const uint ndim
) {
    uint i = get_global_id(0);
    uint index[MAX_NDIM];
    ws_to_index(i, base_strides, ndim, index);
    uint lhs_index = 0;
    uint rhs_index = 0;
    for (uint d = 0; d < ndim; d++) {
        lhs_index += index[d] * lhs_strides[d];
        rhs_index += index[d] * rhs_strides[d];
    }
    output[i] = lhs[lhs_index] + rhs[rhs_index];
}

__kernel void ew_sub_fwd(
    __global const scalar_t* lhs,
    __constant const uint* lhs_strides,
    __global const scalar_t* rhs,
    __constant const uint* rhs_strides,
    __global scalar_t* output,
    __constant const uint* base_strides,
    const uint ndim
) {
    uint i = get_global_id(0);
    uint index[MAX_NDIM];
    ws_to_index(i, base_strides, ndim, index);
    uint lhs_index = 0;
    uint rhs_index = 0;
    for (uint d = 0; d < ndim; d++) {
        lhs_index += index[d] * lhs_strides[d];
        rhs_index += index[d] * rhs_strides[d];
    }
    output[i] = lhs[lhs_index] - rhs[rhs_index];
}

__kernel void ew_mul_fwd(
    __global const scalar_t* lhs,
    __constant const uint* lhs_strides,
    __global const scalar_t* rhs,
    __constant const uint* rhs_strides,
    __global scalar_t* output,
    __constant const uint* base_strides,
    const uint ndim
) {
    uint i = get_global_id(0);
    uint index[MAX_NDIM];
    ws_to_index(i, base_strides, ndim, index);
    uint lhs_index = 0;
    uint rhs_index = 0;
    for (uint d = 0; d < ndim; d++) {
        lhs_index += index[d] * lhs_strides[d];
        rhs_index += index[d] * rhs_strides[d];
    }
    output[i] = lhs[lhs_index] * rhs[rhs_index];
}

__kernel void ew_div_fwd(
    __global const scalar_t* lhs,
    __constant const uint* lhs_strides,
    __global const scalar_t* rhs,
    __constant const uint* rhs_strides,
    __global scalar_t* output,
    __constant const uint* base_strides,
    const uint ndim
) {
    uint i = get_global_id(0);
    uint index[MAX_NDIM];
    ws_to_index(i, base_strides, ndim, index);
    uint lhs_index = 0;
    uint rhs_index = 0;
    for (uint d = 0; d < ndim; d++) {
        lhs_index += index[d] * lhs_strides[d];
        rhs_index += index[d] * rhs_strides[d];
    }
    output[i] = lhs[lhs_index] / rhs[rhs_index];
}

__kernel void unary_neg_fwd(
    __global const scalar_t* input,
    __constant const uint* input_strides,
    __global scalar_t* output,
    __constant const uint* base_strides,
    const uint ndim
) {
    uint i = get_global_id(0);
    uint index[MAX_NDIM];
    ws_to_index(i, base_strides, ndim, index);
    uint input_index = 0;
    for (uint d = 0; d < ndim; d++) {
        input_index += index[d] * input_strides[d];
    }
    output[i] = -input[input_index];
}

__kernel void abs_fwd(
    __global const scalar_t* input,
    __constant const uint* input_strides,
    __global scalar_t* output,
    __constant const uint* base_strides,
    const uint ndim
) {
    uint i = get_global_id(0);
    uint index[MAX_NDIM];
    ws_to_index(i, base_strides, ndim, index);
    uint input_index = 0;
    for (uint d = 0; d < ndim; d++) {
        input_index += index[d] * input_strides[d];
    }
    output[i] = fabs(input[input_index]);
}

__kernel void pow_fwd(
    __global const scalar_t* input,
    __constant const uint* input_strides,
    __global scalar_t* exponent,
    __global scalar_t* output,
    __constant const uint* base_strides,
    const uint ndim
) {
    uint i = get_global_id(0);
    uint index[MAX_NDIM];
    ws_to_index(i, base_strides, ndim, index);
    uint input_index = 0;
    for (uint d = 0; d < ndim; d++) {
        input_index += index[d] * input_strides[d];
    }
    output[i] = pow(input[input_index], exponent[0]);
}

__kernel void sqrt_fwd(
    __global const scalar_t* input,
    __constant const uint* input_strides,
    __global scalar_t* output,
    __constant const uint* base_strides,
    const uint ndim
) {
    uint i = get_global_id(0);
    uint index[MAX_NDIM];
    ws_to_index(i, base_strides, ndim, index);
    uint input_index = 0;
    for (uint d = 0; d < ndim; d++) {
        input_index += index[d] * input_strides[d];
    }
    output[i] = sqrt(input[input_index]);
}

__kernel void exp_fwd(
    __global const scalar_t* input,
    __constant const uint* input_strides,
    __global scalar_t* output,
    __constant const uint* base_strides,
    const uint ndim
) {
    uint i = get_global_id(0);
    uint index[MAX_NDIM];
    ws_to_index(i, base_strides, ndim, index);
    uint input_index = 0;
    for (uint d = 0; d < ndim; d++) {
        input_index += index[d] * input_strides[d];
    }
    output[i] = exp(input[input_index]);
}

__kernel void ln_fwd(
    __global const scalar_t* input,
    __constant const uint* input_strides,
    __global scalar_t* output,
    __constant const uint* base_strides,
    const uint ndim
) {
    uint i = get_global_id(0);
    uint index[MAX_NDIM];
    ws_to_index(i, base_strides, ndim, index);
    uint input_index = 0;
    for (uint d = 0; d < ndim; d++) {
        input_index += index[d] * input_strides[d];
    }
    output[i] = log(input[input_index]);
}

__kernel void log_fwd(
    __global const scalar_t* input,
    __constant const uint* input_strides,
    __global scalar_t* base,
    __global scalar_t* output,
    __constant const uint* base_strides,
    const uint ndim
) {
    uint i = get_global_id(0);
    uint index[MAX_NDIM];
    ws_to_index(i, base_strides, ndim, index);
    uint input_index = 0;
    for (uint d = 0; d < ndim; d++) {
        input_index += index[d] * input_strides[d];
    }
    output[i] = log(input[input_index]) / log(base[0]);
}