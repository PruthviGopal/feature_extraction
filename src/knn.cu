#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>

// Helper functions for CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#include "knn2_struct.h"

//#define const float_relativeTolerance = 1e-6
#define KNN_KERNEL_USE_ITERATION

//struct __device_builtin__ __align__(4) uchar4
//{
//    unsigned char x, y, z, w;
//};

//extern __shared__ float knn_length[];

__device__ void sortKnn2(int& knn_data0, int& knn_data1, float& knn_len0, float& knn_len1) {
    int knn_data_tmp;
    float knn_len_tmp;

    if(knn_len1 < knn_len0) {
        knn_data_tmp = knn_data0;
        knn_len_tmp  = knn_len0;

        knn_data0 = knn_data1;
        knn_len0  = knn_len1;

        knn_data1 = knn_data_tmp;
        knn_len1  = knn_len_tmp ;
    }
}

inline void sortKnn2_c(knn2* one_knn2) {
    int knn_data_tmp;
    float knn_len_tmp;

    if(one_knn2->len1 < one_knn2->len0) {
        knn_data_tmp = one_knn2->data0;
        knn_len_tmp  = one_knn2->len0;

        one_knn2->data0 = one_knn2->data1;
        one_knn2->len0  = one_knn2->len1;

        one_knn2->data1 = knn_data_tmp;
        one_knn2->len1  = knn_len_tmp ;
    }
}

__device__ void sortKnn3(int& knn_data0, int& knn_data1, int& knn_data2, float& knn_len0, float& knn_len1, float& knn_len2) {
    int knn_data_tmp;
    float knn_len_tmp;

    if(knn_len2 < knn_len0) {
        knn_data_tmp = knn_data0;
        knn_len_tmp  = knn_len0;

        knn_data0 = knn_data1;
        knn_len0  = knn_len1;

        knn_data1 = knn_data2;
        knn_len1  = knn_len2;

        knn_data2 = knn_data_tmp;
        knn_len2  = knn_len_tmp ;
    } else if(knn_len2 < knn_len1) {
        knn_data_tmp = knn_data1;
        knn_len_tmp = knn_len1;

        knn_data1 = knn_data2;
        knn_len1  = knn_len2;

        knn_data2 = knn_data_tmp;
        knn_len2  = knn_len_tmp ;
    }

}

inline void sortKnn3_c(knn2* one_knn2) {
    int knn_data_tmp;
    float knn_len_tmp;

    if(one_knn2->len2 < one_knn2->len0) {
        knn_data_tmp = one_knn2->data0;
        knn_len_tmp  = one_knn2->len0;

        one_knn2->data0 = one_knn2->data1;
        one_knn2->len0  = one_knn2->len1;

        one_knn2->data1 = one_knn2->data2;
        one_knn2->len1  = one_knn2->len2;

        one_knn2->data2 = knn_data_tmp;
        one_knn2->len2  = knn_len_tmp ;
    } else if(one_knn2->len2 < one_knn2->len1) {
        knn_data_tmp = one_knn2->data1;
        knn_len_tmp = one_knn2->len1;

        one_knn2->data1 = one_knn2->data2;
        one_knn2->len1  = one_knn2->len2;

        one_knn2->data2 = knn_data_tmp;
        one_knn2->len2  = knn_len_tmp ;
    }

}

__device__ void calcSameLine(
    float* img, int& point_idx, int& knn_data0, int& knn_data1, int& knn_data2, float& knn_len0, float& knn_len1, float& knn_len2,
    bool& line01, bool& line02
) {
    float x0_vec, y0_vec, z0_vec;
    float xDes_vec, yDes_vec, zDes_vec;
    double length;
    const float float_relativeTolerance = 1e-6;

    //calculate point_idx<->knn_data0 & point_idx<->knn_data1 
    x0_vec = *(img + point_idx*3) - *(img + knn_data0*3);
    y0_vec = *(img + point_idx*3+1) - *(img + knn_data0*3+1);
    z0_vec = *(img + point_idx*3+2) - *(img + knn_data0*3+2);
    length = sqrt(pow(x0_vec, 2) + pow(y0_vec, 2) + pow(z0_vec, 2));
    x0_vec = x0_vec / (float)length;
    y0_vec = y0_vec / (float)length;
    z0_vec = z0_vec / (float)length;

    xDes_vec = *(img + point_idx*3) - *(img + knn_data1*3);
    yDes_vec = *(img + point_idx*3+1) - *(img + knn_data1*3+1);
    zDes_vec = *(img + point_idx*3+2) - *(img + knn_data1*3+2);
    length = sqrt(pow(xDes_vec, 2) + pow(yDes_vec, 2) + pow(zDes_vec, 2));
    xDes_vec = xDes_vec / (float)length;
    yDes_vec = yDes_vec / (float)length;
    zDes_vec = zDes_vec / (float)length;

    //point_idx<->knn_data0 & point_idx<->knn_data1
    if(
        abs(x0_vec - xDes_vec) <= float_relativeTolerance && abs(y0_vec - yDes_vec) <= float_relativeTolerance && 
        abs(z0_vec - zDes_vec) <= float_relativeTolerance
    ) {
        line01 = true;
    }

    xDes_vec = *(img + point_idx*3) - *(img + knn_data2*3);
    yDes_vec = *(img + point_idx*3+1) - *(img + knn_data2*3+1);
    zDes_vec = *(img + point_idx*3+2) - *(img + knn_data2*3+2);
    length = sqrt(pow(xDes_vec, 2) + pow(yDes_vec, 2) + pow(zDes_vec, 2));
    xDes_vec = xDes_vec / (float)length;
    yDes_vec = yDes_vec / (float)length;
    zDes_vec = zDes_vec / (float)length;

    //point_idx<->knn_data0 & point_idx<->knn_data2
    if(
        abs(x0_vec - xDes_vec) <= float_relativeTolerance && abs(y0_vec - yDes_vec) <= float_relativeTolerance && 
        abs(z0_vec - zDes_vec) <= float_relativeTolerance
    ) {
        line02 = true;
    }
}

inline void calcSameLine_c(float* img, int& point_idx, knn2* one_knn2) {
    float x0_vec, y0_vec, z0_vec;
    float xDes_vec, yDes_vec, zDes_vec;
    double length;
    const float float_relativeTolerance = 1e-6;

    //calculate point_idx<->knn_data0 & point_idx<->one_knn2->data1 
    x0_vec = *(img + point_idx*3) - *(img + one_knn2->data0*3);
    y0_vec = *(img + point_idx*3+1) - *(img + one_knn2->data0*3+1);
    z0_vec = *(img + point_idx*3+2) - *(img + one_knn2->data0*3+2);
    length = sqrt(pow(x0_vec, 2) + pow(y0_vec, 2) + pow(z0_vec, 2));
    x0_vec = x0_vec / (float)length;
    y0_vec = y0_vec / (float)length;
    z0_vec = z0_vec / (float)length;

    xDes_vec = *(img + point_idx*3) - *(img + one_knn2->data1*3);
    yDes_vec = *(img + point_idx*3+1) - *(img + one_knn2->data1*3+1);
    zDes_vec = *(img + point_idx*3+2) - *(img + one_knn2->data1*3+2);
    length = sqrt(pow(xDes_vec, 2) + pow(yDes_vec, 2) + pow(zDes_vec, 2));
    xDes_vec = xDes_vec / (float)length;
    yDes_vec = yDes_vec / (float)length;
    zDes_vec = zDes_vec / (float)length;

    //point_idx<->one_knn2->data0 & point_idx<->one_knn2->data1
    if(
        abs(x0_vec - xDes_vec) <= float_relativeTolerance && abs(y0_vec - yDes_vec) <= float_relativeTolerance && 
        abs(z0_vec - zDes_vec) <= float_relativeTolerance
    ) {
        one_knn2->line01 = true;
    }

    xDes_vec = *(img + point_idx*3) - *(img + one_knn2->data2*3);
    yDes_vec = *(img + point_idx*3+1) - *(img + one_knn2->data2*3+1);
    zDes_vec = *(img + point_idx*3+2) - *(img + one_knn2->data2*3+2);
    length = sqrt(pow(xDes_vec, 2) + pow(yDes_vec, 2) + pow(zDes_vec, 2));
    xDes_vec = xDes_vec / (float)length;
    yDes_vec = yDes_vec / (float)length;
    zDes_vec = zDes_vec / (float)length;

    //point_idx<->one_knn2->data0 & point_idx<->one_knn2->data2
    if(
        abs(x0_vec - xDes_vec) <= float_relativeTolerance && abs(y0_vec - yDes_vec) <= float_relativeTolerance && 
        abs(z0_vec - zDes_vec) <= float_relativeTolerance
    ) {
        one_knn2->line02 = true;
    }
}

__global__ void vecKnnKernel(float* img, int* knn_data, int n) {

    // Calculate global thread index based on the block and thread indices ----
    int point_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(point_idx >= n) {
        return;
    }

    float* curr_img_idx = img + point_idx * 3;
    double length;
    int knn_data0, knn_data1, knn_data2;
    float knn_len0, knn_len1, knn_len2;

    //initialize the first knn
    //for point 0 and point 1, the initial 2-nn points are 2 and 3
    //for points 2~inf, the initial 2-nn points are 0 and 1
    if(point_idx < 2) {
        length  = pow(*curr_img_idx - *(img + 6), 2);
        length += pow(*(curr_img_idx + 1) - *(img + 7), 2);
        length += pow(*(curr_img_idx + 2) - *(img + 8), 2);
        knn_data0 = 2;
        knn_len0 = length;

        length  = pow(*curr_img_idx - *(img + 9), 2);
        length += pow(*(curr_img_idx + 1) - *(img + 10), 2);
        length += pow(*(curr_img_idx + 2) - *(img + 11), 2);

        knn_data1 = 3;
        knn_len1 = length;
    }
    else {
        length  = pow(*curr_img_idx - *img, 2);
        length += pow(*(curr_img_idx + 1) - *(img + 1), 2);
        length += pow(*(curr_img_idx + 2) - *(img + 2), 2);
        knn_data0 = 0;
        knn_len0 = length;

        length  = pow(*curr_img_idx - *(img + 3), 2);
        length += pow(*(curr_img_idx + 1) - *(img + 4), 2);
        length += pow(*(curr_img_idx + 2) - *(img + 5), 2);
        knn_data1 = 1;
        knn_len1 = length;
    }

    //sort the knn
    sortKnn2(knn_data0, knn_data1, knn_len0, knn_len1);

    for(int i=0; i<n; i++) {
        if(point_idx == i) {
            continue;
        }
        
        //get the point i
        //calculate length of point i
        length  = pow(*curr_img_idx - *(img + i*3), 2);
        length += pow(*(curr_img_idx + 1) - *(img + i*3 + 1), 2);
        length += pow(*(curr_img_idx + 2) - *(img + i*3 + 2), 2);

        //calculate length point_idx->i
        knn_data2 = i;
        knn_len2 = length;

        //reorder point sequence based on length
        sortKnn3(knn_data0, knn_data1, knn_data2, knn_len0, knn_len1, knn_len2);

        //check point_idx<->knn_data0 & point_idx<->knn_data1, 
        //point_idx<->knn_data0 & point_idx<->knn_data2, point_idx<->knn_data1 & point_idx<->knn_data2 
        //are on the same line
        //line01 is if lines line0 (point_idx<->knn_data0) & line1 (point_idx<->knn_data1) are on the same line
        //line02 is if lines line0 (point_idx<->knn_data0) & line2 (point_idx<->knn_data2) are on the same line
        bool line01 = false;
        bool line02 = false; 
        calcSameLine(img, point_idx, knn_data0, knn_data1, knn_data2, knn_len0, knn_len1, knn_len2, line01, line02);

        //remove the point which if farthest or on the same line
        //if(line01 == false) {
        //    //knn_data0 and knn_data1 are not on the same line
        //    //ok
        //} else if(line02 = false) {
        //    //knn_data0 and knn_data1 are on the same line
        //    //knn_data0 and knn_data2 are not on the same line
        //    //knn_data2 -> knn_data1
        //} 
        if(line01 == true && line02 == false) {
            knn_data1 = knn_data2;
            knn_len1 = knn_len2;
        }

    }

    *(knn_data + point_idx*2)= knn_data0;
    *(knn_data + point_idx*2 + 1) = knn_data1;

}

__global__ void vecKnnKernel_iter_init(float* img, int* knn_data, float* knn_len, int n) {

    // Calculate global thread index based on the block and thread indices ----
    int point_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(point_idx >= n) {
        return;
    }

    float* curr_img_idx = img + point_idx * 3;
    double length;
    int knn_data0, knn_data1;
    float knn_len0, knn_len1;

    //initialize the first knn
    //for point 0 and point 1, the initial 2-nn points are 2 and 3
    //for points 2~inf, the initial 2-nn points are 0 and 1
    if(point_idx < 2) {
        length  = pow(*curr_img_idx - *(img + 6), 2);
        length += pow(*(curr_img_idx + 1) - *(img + 7), 2);
        length += pow(*(curr_img_idx + 2) - *(img + 8), 2);
        knn_data0 = 2;
        knn_len0 = length;

        length  = pow(*curr_img_idx - *(img + 9), 2);
        length += pow(*(curr_img_idx + 1) - *(img + 10), 2);
        length += pow(*(curr_img_idx + 2) - *(img + 11), 2);

        knn_data1 = 3;
        knn_len1 = length;
    }
    else {
        length  = pow(*curr_img_idx - *img, 2);
        length += pow(*(curr_img_idx + 1) - *(img + 1), 2);
        length += pow(*(curr_img_idx + 2) - *(img + 2), 2);
        knn_data0 = 0;
        knn_len0 = length;

        length  = pow(*curr_img_idx - *(img + 3), 2);
        length += pow(*(curr_img_idx + 1) - *(img + 4), 2);
        length += pow(*(curr_img_idx + 2) - *(img + 5), 2);
        knn_data1 = 1;
        knn_len1 = length;
    }

    //sort the knn
    sortKnn2(knn_data0, knn_data1, knn_len0, knn_len1);

    *(knn_data + point_idx*2)= knn_data0;
    *(knn_data + point_idx*2 + 1) = knn_data1;
    *(knn_len + point_idx*2)= knn_len0;
    *(knn_len + point_idx*2 + 1) = knn_len1;

}


__global__ void vecKnnKernel_iter(float* img, int* knn_data, float* knn_len, int n, int iter_begin, int iter_end) {

    // Calculate global thread index based on the block and thread indices ----
    int point_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(point_idx >= n) {
        return;
    }

    float* curr_img_idx = img + point_idx * 3;
    double length;
    int knn_data0, knn_data1, knn_data2;
    float knn_len0, knn_len1, knn_len2;

    knn_data0 = *(knn_data + point_idx*2);
    knn_data1 = *(knn_data + point_idx*2 + 1);
    knn_len0  = *(knn_len  + point_idx*2);
    knn_len1  = *(knn_len  + point_idx*2 + 1);

    for(int i=iter_begin; i<iter_end; i++) {
        if(point_idx == i) {
            continue;
        }
        
        //get the point i
        //calculate length of point i
        length  = pow(*curr_img_idx - *(img + i*3), 2);
        length += pow(*(curr_img_idx + 1) - *(img + i*3 + 1), 2);
        length += pow(*(curr_img_idx + 2) - *(img + i*3 + 2), 2);

        //calculate length point_idx->i
        knn_data2 = i;
        knn_len2 = length;

        //reorder point sequence based on length
        sortKnn3(knn_data0, knn_data1, knn_data2, knn_len0, knn_len1, knn_len2);

        //check point_idx<->knn_data0 & point_idx<->knn_data1, 
        //point_idx<->knn_data0 & point_idx<->knn_data2, point_idx<->knn_data1 & point_idx<->knn_data2 
        //are on the same line
        //line01 is if lines line0 (point_idx<->knn_data0) & line1 (point_idx<->knn_data1) are on the same line
        //line02 is if lines line0 (point_idx<->knn_data0) & line2 (point_idx<->knn_data2) are on the same line
        bool line01 = false;
        bool line02 = false; 
        calcSameLine(img, point_idx, knn_data0, knn_data1, knn_data2, knn_len0, knn_len1, knn_len2, line01, line02);

        //remove the point which if farthest or on the same line
        //if(line01 == false) {
        //    //knn_data0 and knn_data1 are not on the same line
        //    //ok
        //} else if(line02 = false) {
        //    //knn_data0 and knn_data1 are on the same line
        //    //knn_data0 and knn_data2 are not on the same line
        //    //knn_data2 -> knn_data1
        //} 
        if(line01 == true && line02 == false) {
            knn_data1 = knn_data2;
            knn_len1 = knn_len2;
        }

    }

    *(knn_data + point_idx*2)= knn_data0;
    *(knn_data + point_idx*2 + 1) = knn_data1;
    *(knn_len + point_idx*2)= knn_len0;
    *(knn_len + point_idx*2 + 1) = knn_len1;

}

bool knn(float* d_img, int* d_knn_data, int point_num) {
    clock_t begin, end;

    //////////////////////////////////////////////////////////////
    //Allocate device variables
    //////////////////////////////////////////////////////////////
#ifdef KNN_KERNEL_USE_ITERATION
    printf("Allocating device variables for d_knn_len...\n"); 
    fflush(stdout);
    begin = clock();

    float *d_knn_len;
    cudaMalloc((void **)&d_knn_len, sizeof(float)*point_num*2);

    checkCudaErrors(cudaDeviceSynchronize());
    end = clock();
    printf("Elapsed: %f seconds\n\n", (double)(end - begin) / CLOCKS_PER_SEC);
#endif

    //////////////////////////////////////////////////////////////
    // Launch kernel
    //////////////////////////////////////////////////////////////
    int BLOCK_SIZE = 256;

    dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(ceil(point_num / (float)BLOCK_SIZE), 1, 1);
    //int knn_length_num = BLOCK_SIZE * 4 * 2;

#ifdef KNN_KERNEL_USE_ITERATION
    printf("Launching kernel of knn_init...\n"); 
    fflush(stdout);
    begin =clock();
    vecKnnKernel_iter_init <<<dimGrid, dimBlock>>>(d_img, d_knn_data, d_knn_len, point_num);
    checkCudaErrors(cudaDeviceSynchronize());
    end = clock();
    printf("Elapsed: %f seconds\n\n", (double)(end - begin) / CLOCKS_PER_SEC);

    printf("Launching kernel of knn...\n"); 
    fflush(stdout);
    begin =clock();

    int inc_num = 900;
    int iter_begin = 0;
    int iter_end = inc_num;
    if(iter_end > point_num) {
        iter_end = point_num;
    }
    while(true) {
        printf("Iter Begin: %d, Iter End: %d\n", iter_begin, iter_end);
        vecKnnKernel_iter <<<dimGrid, dimBlock>>>(d_img, d_knn_data, d_knn_len, point_num, iter_begin, iter_end);

        checkCudaErrors(cudaDeviceSynchronize());

        iter_begin = iter_end;
        iter_end += inc_num;
        if(iter_end > point_num) {
            iter_end = point_num;
        }

        if(iter_begin == iter_end) {
            break;
        }
    }

#else
    printf("Launching kernel of knn...\n"); 
    fflush(stdout);
    begin =clock();

    vecKnnKernel <<<dimGrid, dimBlock>>>(d_img, d_knn_data, point_num);
#endif

    end = clock();
    printf("Elapsed: %f seconds\n\n", (double)(end - begin) / CLOCKS_PER_SEC);

    //////////////////////////////////////////////////////////////
    // Free memory
    //////////////////////////////////////////////////////////////
#ifdef KNN_KERNEL_USE_ITERATION
    cudaFree(d_knn_len);
#endif

    return true;
}

bool knn_c(float* img, knn2 *knn2_arr, int point_num) {
    clock_t begin, end;
    //////////////////////////////////////////////////////////////
    // allocate memory
    //////////////////////////////////////////////////////////////
    float* knn_len = new float[point_num*2];

    //////////////////////////////////////////////////////////////
    // calculate 2-nn
    //////////////////////////////////////////////////////////////
    printf("Running knn_c...\n"); 
    fflush(stdout);
    begin =clock();

    float length;
    for(int i=0; i<point_num; i++) {
        float* curr_img_idx = img + i * 3;
        int ctrl_cnt = 0;

        for(int j=0; j<point_num; j++) {
            if(i==j) {
                continue;
            }
            float* cmp_img_idx = img + j * 3;

            if(ctrl_cnt == 0) {
                length  = pow(*curr_img_idx - *cmp_img_idx, 2);
                length += pow(*(curr_img_idx + 1) - *(cmp_img_idx + 1), 2);
                length += pow(*(curr_img_idx + 2) - *(cmp_img_idx + 2), 2);
                (knn2_arr + i)->data0 = j;
                (knn2_arr + i)->len0 = length;
                ctrl_cnt++;
                continue;
            } else if(ctrl_cnt == 1) {
                length  = pow(*curr_img_idx - *cmp_img_idx, 2);
                length += pow(*(curr_img_idx + 1) - *(cmp_img_idx + 1), 2);
                length += pow(*(curr_img_idx + 2) - *(cmp_img_idx + 2), 2);

                if(length < (knn2_arr + i)->len0) {
                    (knn2_arr + i)->len1 = (knn2_arr + i)->len0;
                    (knn2_arr + i)->data1 = (knn2_arr + i)->data0;
                    (knn2_arr + i)->data0 = j;
                    (knn2_arr + i)->len0 = length;
                } else {
                    (knn2_arr + i)->data1 = j;
                    (knn2_arr + i)->len1 = length;
                }
                ctrl_cnt++;
                continue;
            }

            //get the point j
            //calculate length of point j
            length  = pow(*curr_img_idx - *cmp_img_idx, 2);
            length += pow(*(curr_img_idx + 1) - *(cmp_img_idx + 1), 2);
            length += pow(*(curr_img_idx + 2) - *(cmp_img_idx + 2), 2);

            //calculate length point_idx->i
            (knn2_arr + i)->data2 = j;
            (knn2_arr + i)->len2 = length;

            //reorder point sequence based on length
            sortKnn3_c(knn2_arr + i);

            //check point_idx<->knn_data0 & point_idx<->knn_data1, 
            //point_idx<->knn_data0 & point_idx<->knn_data2, point_idx<->knn_data1 & point_idx<->knn_data2 
            //are on the same line
            //line01 is if lines line0 (point_idx<->knn_data0) & line1 (point_idx<->knn_data1) are on the same line
            //line02 is if lines line0 (point_idx<->knn_data0) & line2 (point_idx<->knn_data2) are on the same line
            (knn2_arr + i)->line01 = false;
            (knn2_arr + i)->line02 = false;
            calcSameLine_c(img, i, knn2_arr + i);

            //remove the point which if farthest or on the same line
            if((knn2_arr + i)->line01 == true && (knn2_arr + i)->line02 == false) {
                (knn2_arr + i)->data1 = (knn2_arr + i)->data2;
                (knn2_arr + i)->len1  = (knn2_arr + i)->len2;
            }

        }

    }

    end = clock();
    printf("Elapsed: %f seconds\n\n", (double)(end - begin) / CLOCKS_PER_SEC);

    return true;
}


