#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cstdio>
__global__ void probe(float *out) {
 const float v=1.5f;
 const unsigned e=__float_as_uint(v)&0x7f800000u;
 const float inv=__uint_as_float(0x7f000000u-e), scale=__uint_as_float(e);
 unsigned short p;
 float packed_input=v*inv*6.0f;
 asm("{ .reg .b8 __t; cvt.rn.satfinite.e2m1x2.f32 __t, %2, %1; mov.b16 %0, {__t, 0}; }" : "=h"(p):"f"(packed_input),"f"(0.0f));
 __nv_fp4_e2m1 q;q.__x=p&15;
 out[0]=v;out[1]=packed_input;out[2]=(float)q;out[3]=(float)q*scale/6.0f;
 const float safe_inv=1.0f/v;
 __nv_fp4_e2m1 safe(v*safe_inv*6.0f);
 out[4]=(float)safe*v/6.0f;
}
int main(){float *p,h[5];if(cudaMalloc(&p,sizeof(h))!=cudaSuccess)return 1;probe<<<1,1>>>(p);if(cudaDeviceSynchronize()!=cudaSuccess)return 2;if(cudaMemcpy(h,p,sizeof(h),cudaMemcpyDeviceToHost)!=cudaSuccess)return 3;cudaFree(p);printf("input=%.2f current_quantizer_input=%.2f fp4_value=%.2f current_reconstruction=%.2f exact_max_scaling_reconstruction=%.2f\n",h[0],h[1],h[2],h[3],h[4]);return h[3]==1.0f&&h[4]==1.5f?0:4;}
