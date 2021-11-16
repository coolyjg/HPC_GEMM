const char* sgemm_desc = "Simple blocked sgemm with intrinsic 8*8 packed A";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 32
#endif

#define min(a,b) (((a)<(b))?(a):(b))

#include "arm_neon.h"

/* This auxiliary subroutine performs a smaller sgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
//基础代码
static void do_block (int lda, int M, int N, int K, float* A, float* B, float* C)
{
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
    /* For each column j of B */ 
    for (int j = 0; j < N; ++j) 
    {
      /* Compute C(i,j) */
      float cij = C[i+j*lda];
      for (int k = 0; k < K; ++k)
    	cij += A[i+k*lda] * B[k+j*lda];
      C[i+j*lda] = cij;
    }
}

//3.3. 每次操作4个元素
static void do_block_intrinsic (int lda, int M, int N, int K, float* A, float* B, float* C){
    for(int j = 0; j< N; j++) {//按列计算
        int i;
        for(i = 0; i< M-M%4; i+=4){
            float32x4_t buf = vld1q_f32(&C[i+j*lda]);
            for(int k = 0; k<K; k++){
                float32x4_t a = vld1q_f32(&A[i+k*lda]);
                register float b = B[k+j*lda];
                buf = vmlaq_n_f32(buf, a, b);
            }
            vst1q_f32(&C[i+j*lda], buf);
        }
        for(; i< M; i++){
            register float t = C[i+j*lda];
            for(int k = 0; k<K; k++){
                t+=A[i+k*lda]*B[k+j*lda];
            }
            C[i+j*lda] = t;
        }
    }
}

//3.4 扩展到4*4
void intrinsic_4x4(int lda, float* A, float* B, float* C){ //用寄存器来计算一个4x4的方格
    float *b_0, *b_1, *b_2, *b_3;
    b_0 = B;
    b_1 = &B[0+lda*1];
    b_2 = &B[0+lda*2];
    b_3 = &B[0+lda*3];
    float32x4_t c_0 = {0};
    float32x4_t c_1 = {0};
    float32x4_t c_2 = {0};
    float32x4_t c_3 = {0};
    register float b_0_reg, b_1_reg, b_2_reg, b_3_reg;
    for(int k = 0; k<lda; k++){
        float32x4_t a = vld1q_f32(&A[0+k*lda]);
        b_0_reg = *b_0;
        b_1_reg = *b_1;
        b_2_reg = *b_2;
        b_3_reg = *b_3;
        c_0 = vmlaq_n_f32(c_0, a, b_0_reg);
        c_1 = vmlaq_n_f32(c_1, a, b_1_reg);
        c_2 = vmlaq_n_f32(c_2, a, b_2_reg);
        c_3 = vmlaq_n_f32(c_3, a, b_3_reg);
        b_0++;
        b_1++;
        b_2++;
        b_3++;
    }
    float32x4_t c_st = vld1q_f32(C);
    c_st = vaddq_f32(c_st, c_0);
    vst1q_f32(C, c_st);

    c_st = vld1q_f32(&C[lda]);
    c_st = vaddq_f32(c_st, c_1);
    vst1q_f32(&C[lda], c_st);

    c_st = vld1q_f32(&C[lda*2]);
    c_st = vaddq_f32(c_st, c_2);
    vst1q_f32(&C[lda*2], c_st);

    c_st = vld1q_f32(&C[lda*3]);
    c_st = vaddq_f32(c_st, c_3);
    vst1q_f32(&C[lda*3], c_st);
}

void do_block_intrinsic_ext(int lda, float* A, float* B, float* C){ 
    int i, j;
    for(j = 0; j<((lda)&(~3)); j+=4){
        for(i = 0; i<((lda)&(~3)); i+=4){
            intrinsic_4x4(lda, &A[i], &B[j*lda], &C[i+j*lda]);
        }
        for(; i<lda; i++){
            register float c_0, c_1, c_2, c_3;
            c_0 = C[i+j*lda];
            c_1 = C[i+(j+1)*lda];
            c_2 = C[i+(j+2)*lda];
            c_3 = C[i+(j+3)*lda];
            for(int k = 0; k<lda; k++){
                c_0 += A[i+k*lda]*B[k+j*lda];
                c_1 += A[i+k*lda]*B[k+(j+1)*lda];
                c_2 += A[i+k*lda]*B[k+(j+2)*lda];
                c_3 += A[i+k*lda]*B[k+(j+3)*lda];
            }
            C[i+j*lda] = c_0;
            C[i+(j+1)*lda] = c_1;
            C[i+(j+2)*lda] = c_2;
            C[i+(j+3)*lda] = c_3;
        }
    }
    for(; j< lda; j++){
        for(i = 0; i<((lda)&(~3)); i+=4){
            float32x4_t buf = vld1q_f32(&C[i+j*lda]);
            for(int k = 0; k< lda; k++){
                float32x4_t a = vld1q_f32(&A[i+k*lda]);
                register float b_reg = B[k+j*lda];
                buf = vmlaq_n_f32(buf, a, b_reg);
            }
            vst1q_f32(&C[i+j*lda], buf);
        }
        for(; i<lda; i++){
            float t = C[i+j*lda];
            for(int k = 0; k<lda; k++){
                t += A[i+k*lda]*B[k+j*lda];
            }
            C[i+j*lda] = t;
        }
    }
}

//3.5 扩展到8*8
void intrinsic_8x8(int lda, float* A, float* B, float* C){
    float *b00, *b01, *b02, *b03, *b04, *b05, *b06, *b07;
    b00 = B;
    b01 = &B[lda*1];
    b02 = &B[lda*2];
    b03 = &B[lda*3];
    b04 = &B[lda*4];
    b05 = &B[lda*5];
    b06 = &B[lda*6];
    b07 = &B[lda*7];
    float32x4_t c00 = {0};
    float32x4_t c01 = {0};
    float32x4_t c02 = {0};
    float32x4_t c03 = {0};
    float32x4_t c04 = {0};
    float32x4_t c05 = {0};
    float32x4_t c06 = {0};
    float32x4_t c07 = {0};
    float32x4_t c10 = {0};
    float32x4_t c11 = {0};
    float32x4_t c12 = {0};
    float32x4_t c13 = {0};
    float32x4_t c14 = {0};
    float32x4_t c15 = {0};
    float32x4_t c16 = {0};
    float32x4_t c17 = {0};
    register float b0_reg, b1_reg, b2_reg, b3_reg;
    register float b4_reg, b5_reg, b6_reg, b7_reg;
    for(int k = 0; k<lda; k++){
        float32x4_t a0 = vld1q_f32(&A[k*lda]);
        float32x4_t a1 = vld1q_f32(&A[4+k*lda]);
        b0_reg = *b00;
        b1_reg = *b01;
        b2_reg = *b02;
        b3_reg = *b03;
        b4_reg = *b04;
        b5_reg = *b05;
        b6_reg = *b06;
        b7_reg = *b07;
        c00 = vmlaq_n_f32(c00, a0, b0_reg);
        c01 = vmlaq_n_f32(c01, a0, b1_reg);
        c02 = vmlaq_n_f32(c02, a0, b2_reg);
        c03 = vmlaq_n_f32(c03, a0, b3_reg);
        c04 = vmlaq_n_f32(c04, a0, b4_reg);
        c05 = vmlaq_n_f32(c05, a0, b5_reg);
        c06 = vmlaq_n_f32(c06, a0, b6_reg);
        c07 = vmlaq_n_f32(c07, a0, b7_reg);
        c10 = vmlaq_n_f32(c10, a1, b0_reg);
        c11 = vmlaq_n_f32(c11, a1, b1_reg);
        c12 = vmlaq_n_f32(c12, a1, b2_reg);
        c13 = vmlaq_n_f32(c13, a1, b3_reg);
        c14 = vmlaq_n_f32(c14, a1, b4_reg);
        c15 = vmlaq_n_f32(c15, a1, b5_reg);
        c16 = vmlaq_n_f32(c16, a1, b6_reg);
        c17 = vmlaq_n_f32(c17, a1, b7_reg);
        b00++;
        b01++;
        b02++;
        b03++;
        b04++;
        b05++;
        b06++;
        b07++;
    }
    float32x4_t c_st = vld1q_f32(C);
    c_st = vaddq_f32(c_st, c00);
    vst1q_f32(C, c_st);

    c_st = vld1q_f32(&C[lda*1]);
    c_st = vaddq_f32(c_st, c01);
    vst1q_f32(&C[lda*1], c_st);

    c_st = vld1q_f32(&C[lda*2]);
    c_st = vaddq_f32(c_st, c02);
    vst1q_f32(&C[lda*2], c_st);

    c_st = vld1q_f32(&C[lda*3]);
    c_st = vaddq_f32(c_st, c03);
    vst1q_f32(&C[lda*3], c_st);

    c_st = vld1q_f32(&C[lda*4]);
    c_st = vaddq_f32(c_st, c04);
    vst1q_f32(&C[lda*4], c_st);

    c_st = vld1q_f32(&C[lda*5]);
    c_st = vaddq_f32(c_st, c05);
    vst1q_f32(&C[lda*5], c_st);

    c_st = vld1q_f32(&C[lda*6]);
    c_st = vaddq_f32(c_st, c06);
    vst1q_f32(&C[lda*6], c_st);

    c_st = vld1q_f32(&C[lda*7]);
    c_st = vaddq_f32(c_st, c07);
    vst1q_f32(&C[lda*7], c_st);

    c_st = vld1q_f32(&C[4+lda*0]);
    c_st = vaddq_f32(c_st, c10);
    vst1q_f32(&C[4+lda*0], c_st);

    c_st = vld1q_f32(&C[4+lda*1]);
    c_st = vaddq_f32(c_st, c11);
    vst1q_f32(&C[4+lda*1], c_st);

    c_st = vld1q_f32(&C[4+lda*2]);
    c_st = vaddq_f32(c_st, c12);
    vst1q_f32(&C[4+lda*2], c_st);

    c_st = vld1q_f32(&C[4+lda*3]);
    c_st = vaddq_f32(c_st, c13);
    vst1q_f32(&C[4+lda*3], c_st);

    c_st = vld1q_f32(&C[4+lda*4]);
    c_st = vaddq_f32(c_st, c14);
    vst1q_f32(&C[4+lda*4], c_st);

    c_st = vld1q_f32(&C[4+lda*5]);
    c_st = vaddq_f32(c_st, c15);
    vst1q_f32(&C[4+lda*5], c_st);

    c_st = vld1q_f32(&C[4+lda*6]);
    c_st = vaddq_f32(c_st, c16);
    vst1q_f32(&C[4+lda*6], c_st);

    c_st = vld1q_f32(&C[4+lda*7]);
    c_st = vaddq_f32(c_st, c17);
    vst1q_f32(&C[4+lda*7], c_st);
}

void do_block_intrinsic_ext_8(int lda, float* A, float* B, float* C){ 
    int i, j;
    for(j = 0; j<((lda)&(~7)); j+=8){
        for(i = 0; i<((lda)&(~7)); i+=8){
            intrinsic_8x8(lda, &A[i], &B[j*lda], &C[i+j*lda]);
        }
        for(; i<lda; i++){
            register float c_0, c_1, c_2, c_3;
            register float c_4, c_5, c_6, c_7;
            c_0 = C[i+j*lda];
            c_1 = C[i+(j+1)*lda];
            c_2 = C[i+(j+2)*lda];
            c_3 = C[i+(j+3)*lda];
            c_4 = C[i+(j+4)*lda];
            c_5 = C[i+(j+5)*lda];
            c_6 = C[i+(j+6)*lda];
            c_7 = C[i+(j+7)*lda];
            for(int k = 0; k<lda; k++){
                c_0 += A[i+k*lda]*B[k+j*lda];
                c_1 += A[i+k*lda]*B[k+(j+1)*lda];
                c_2 += A[i+k*lda]*B[k+(j+2)*lda];
                c_3 += A[i+k*lda]*B[k+(j+3)*lda];
                c_4 += A[i+k*lda]*B[k+(j+4)*lda];
                c_5 += A[i+k*lda]*B[k+(j+5)*lda];
                c_6 += A[i+k*lda]*B[k+(j+6)*lda];
                c_7 += A[i+k*lda]*B[k+(j+7)*lda];
            }
            C[i+j*lda] = c_0;
            C[i+(j+1)*lda] = c_1;
            C[i+(j+2)*lda] = c_2;
            C[i+(j+3)*lda] = c_3;
            C[i+(j+4)*lda] = c_4;
            C[i+(j+5)*lda] = c_5;
            C[i+(j+6)*lda] = c_6;
            C[i+(j+7)*lda] = c_7;
        }
    }
    for(; j< lda; j++){
        for(i = 0; i<((lda)&(~7)); i+=8){
            float32x4_t buf = vld1q_f32(&C[i+j*lda]);
            float32x4_t buf2 = vld1q_f32(&C[i+4+j*lda]);
            for(int k = 0; k< lda; k++){
                float32x4_t a = vld1q_f32(&A[i+k*lda]);
                float32x4_t a2 = vld1q_f32(&A[i+4+k*lda]);
                register float b_reg = B[k+j*lda];
                buf = vmlaq_n_f32(buf, a, b_reg);
                buf2 = vmlaq_n_f32(buf2, a2, b_reg);
            }
            vst1q_f32(&C[i+j*lda], buf);
            vst1q_f32(&C[i+4+j*lda], buf2);
        }
        for(; i<lda; i++){
            float t = C[i+j*lda];
            for(int k = 0; k<lda; k++){
                t += A[i+k*lda]*B[k+j*lda];
            }
            C[i+j*lda] = t;
        }
    }
}

//3.6 把A进行pack
void pack_A(int lda, float* A, float* p_a){
    for(int j = 0; j<lda; j++){
        float* temp = &A[0+j*lda];
        *p_a++ = *temp;
        *p_a++ = *(temp+1);
        *p_a++ = *(temp+2);
        *p_a++ = *(temp+3);
        *p_a++ = *(temp+4);
        *p_a++ = *(temp+5);
        *p_a++ = *(temp+6);
        *p_a++ = *(temp+7);
    }
}

void intrinsic_8x8_packA(int lda, float* A, float* B, float* C){
    float *b00, *b01, *b02, *b03, *b04, *b05, *b06, *b07;
    b00 = B;
    b01 = &B[lda*1];
    b02 = &B[lda*2];
    b03 = &B[lda*3];
    b04 = &B[lda*4];
    b05 = &B[lda*5];
    b06 = &B[lda*6];
    b07 = &B[lda*7];
    float32x4_t c00 = {0};
    float32x4_t c01 = {0};
    float32x4_t c02 = {0};
    float32x4_t c03 = {0};
    float32x4_t c04 = {0};
    float32x4_t c05 = {0};
    float32x4_t c06 = {0};
    float32x4_t c07 = {0};
    float32x4_t c10 = {0};
    float32x4_t c11 = {0};
    float32x4_t c12 = {0};
    float32x4_t c13 = {0};
    float32x4_t c14 = {0};
    float32x4_t c15 = {0};
    float32x4_t c16 = {0};
    float32x4_t c17 = {0};
    register float b0_reg, b1_reg, b2_reg, b3_reg;
    register float b4_reg, b5_reg, b6_reg, b7_reg;
    for(int k = 0; k<lda; k++){
        float32x4_t a0 = vld1q_f32(&A[k*8]);
        float32x4_t a1 = vld1q_f32(&A[4+k*8]);
        b0_reg = *b00;
        b1_reg = *b01;
        b2_reg = *b02;
        b3_reg = *b03;
        b4_reg = *b04;
        b5_reg = *b05;
        b6_reg = *b06;
        b7_reg = *b07;
        c00 = vmlaq_n_f32(c00, a0, b0_reg);
        c01 = vmlaq_n_f32(c01, a0, b1_reg);
        c02 = vmlaq_n_f32(c02, a0, b2_reg);
        c03 = vmlaq_n_f32(c03, a0, b3_reg);
        c04 = vmlaq_n_f32(c04, a0, b4_reg);
        c05 = vmlaq_n_f32(c05, a0, b5_reg);
        c06 = vmlaq_n_f32(c06, a0, b6_reg);
        c07 = vmlaq_n_f32(c07, a0, b7_reg);
        c10 = vmlaq_n_f32(c10, a1, b0_reg);
        c11 = vmlaq_n_f32(c11, a1, b1_reg);
        c12 = vmlaq_n_f32(c12, a1, b2_reg);
        c13 = vmlaq_n_f32(c13, a1, b3_reg);
        c14 = vmlaq_n_f32(c14, a1, b4_reg);
        c15 = vmlaq_n_f32(c15, a1, b5_reg);
        c16 = vmlaq_n_f32(c16, a1, b6_reg);
        c17 = vmlaq_n_f32(c17, a1, b7_reg);
        b00++;
        b01++;
        b02++;
        b03++;
        b04++;
        b05++;
        b06++;
        b07++;
    }
    float32x4_t c_st = vld1q_f32(C);
    c_st = vaddq_f32(c_st, c00);
    vst1q_f32(C, c_st);

    c_st = vld1q_f32(&C[lda*1]);
    c_st = vaddq_f32(c_st, c01);
    vst1q_f32(&C[lda*1], c_st);

    c_st = vld1q_f32(&C[lda*2]);
    c_st = vaddq_f32(c_st, c02);
    vst1q_f32(&C[lda*2], c_st);

    c_st = vld1q_f32(&C[lda*3]);
    c_st = vaddq_f32(c_st, c03);
    vst1q_f32(&C[lda*3], c_st);

    c_st = vld1q_f32(&C[lda*4]);
    c_st = vaddq_f32(c_st, c04);
    vst1q_f32(&C[lda*4], c_st);

    c_st = vld1q_f32(&C[lda*5]);
    c_st = vaddq_f32(c_st, c05);
    vst1q_f32(&C[lda*5], c_st);

    c_st = vld1q_f32(&C[lda*6]);
    c_st = vaddq_f32(c_st, c06);
    vst1q_f32(&C[lda*6], c_st);

    c_st = vld1q_f32(&C[lda*7]);
    c_st = vaddq_f32(c_st, c07);
    vst1q_f32(&C[lda*7], c_st);

    c_st = vld1q_f32(&C[4+lda*0]);
    c_st = vaddq_f32(c_st, c10);
    vst1q_f32(&C[4+lda*0], c_st);

    c_st = vld1q_f32(&C[4+lda*1]);
    c_st = vaddq_f32(c_st, c11);
    vst1q_f32(&C[4+lda*1], c_st);

    c_st = vld1q_f32(&C[4+lda*2]);
    c_st = vaddq_f32(c_st, c12);
    vst1q_f32(&C[4+lda*2], c_st);

    c_st = vld1q_f32(&C[4+lda*3]);
    c_st = vaddq_f32(c_st, c13);
    vst1q_f32(&C[4+lda*3], c_st);

    c_st = vld1q_f32(&C[4+lda*4]);
    c_st = vaddq_f32(c_st, c14);
    vst1q_f32(&C[4+lda*4], c_st);

    c_st = vld1q_f32(&C[4+lda*5]);
    c_st = vaddq_f32(c_st, c15);
    vst1q_f32(&C[4+lda*5], c_st);

    c_st = vld1q_f32(&C[4+lda*6]);
    c_st = vaddq_f32(c_st, c16);
    vst1q_f32(&C[4+lda*6], c_st);

    c_st = vld1q_f32(&C[4+lda*7]);
    c_st = vaddq_f32(c_st, c17);
    vst1q_f32(&C[4+lda*7], c_st);
}

void do_block_intrinsic_ext_8_packa(int lda, float* A, float* B, float* C){ 
    int i, j;
    float pack_a[((lda)&(~7))*lda];
    for(j = 0; j<((lda)&(~7)); j+=8){
        for(i = 0; i<((lda)&(~7)); i+=8){
            if(j==0) pack_A(lda, &A[i], &pack_a[i*lda]);
            intrinsic_8x8_packA(lda, &pack_a[i*lda], &B[j*lda], &C[i+j*lda]);
        }
        for(; i<lda; i++){
            register float c_0, c_1, c_2, c_3;
            register float c_4, c_5, c_6, c_7;
            c_0 = C[i+j*lda];
            c_1 = C[i+(j+1)*lda];
            c_2 = C[i+(j+2)*lda];
            c_3 = C[i+(j+3)*lda];
            c_4 = C[i+(j+4)*lda];
            c_5 = C[i+(j+5)*lda];
            c_6 = C[i+(j+6)*lda];
            c_7 = C[i+(j+7)*lda];
            for(int k = 0; k<lda; k++){
                c_0 += A[i+k*lda]*B[k+j*lda];
                c_1 += A[i+k*lda]*B[k+(j+1)*lda];
                c_2 += A[i+k*lda]*B[k+(j+2)*lda];
                c_3 += A[i+k*lda]*B[k+(j+3)*lda];
                c_4 += A[i+k*lda]*B[k+(j+4)*lda];
                c_5 += A[i+k*lda]*B[k+(j+5)*lda];
                c_6 += A[i+k*lda]*B[k+(j+6)*lda];
                c_7 += A[i+k*lda]*B[k+(j+7)*lda];
            }
            C[i+j*lda] = c_0;
            C[i+(j+1)*lda] = c_1;
            C[i+(j+2)*lda] = c_2;
            C[i+(j+3)*lda] = c_3;
            C[i+(j+4)*lda] = c_4;
            C[i+(j+5)*lda] = c_5;
            C[i+(j+6)*lda] = c_6;
            C[i+(j+7)*lda] = c_7;
        }
    }
    for(; j< lda; j++){
        for(i = 0; i<((lda)&(~7)); i+=8){
            float32x4_t buf = vld1q_f32(&C[i+j*lda]);
            float32x4_t buf2 = vld1q_f32(&C[i+4+j*lda]);
            for(int k = 0; k< lda; k++){
                float32x4_t a = vld1q_f32(&A[i+k*lda]);
                float32x4_t a2 = vld1q_f32(&A[i+4+k*lda]);
                register float b_reg = B[k+j*lda];
                buf = vmlaq_n_f32(buf, a, b_reg);
                buf2 = vmlaq_n_f32(buf2, a2, b_reg);
            }
            vst1q_f32(&C[i+j*lda], buf);
            vst1q_f32(&C[i+4+j*lda], buf2);
        }
        for(; i<lda; i++){
            float t = C[i+j*lda];
            for(int k = 0; k<lda; k++){
                t += A[i+k*lda]*B[k+j*lda];
            }
            C[i+j*lda] = t;
        }
    }
}




/* This routine performs a sgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_sgemm (int lda, float* A, float* B, float* C)
{
//   for (int i = 0; i < lda; i += BLOCK_SIZE)
//     for (int j = 0; j < lda; j += BLOCK_SIZE)
//       for (int k = 0; k < lda; k += BLOCK_SIZE)
//       {
// 	int M = min (BLOCK_SIZE, lda-i);
// 	int N = min (BLOCK_SIZE, lda-j);
// 	int K = min (BLOCK_SIZE, lda-k);

	// do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);     //******************基础代码
    // do_block_intrinsic(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda); //************3.3
//       }                                                             //3.4以后的代码取消了基础分块
    // do_block_intrinsic_ext(lda, A, B, C);                          //*****************************3.4
    // do_block_intrinsic_ext_8(lda, A, B, C);                        //*****************************3.5
    do_block_intrinsic_ext_8_packa(lda, A, B, C);                     //*****************************3.6
}

