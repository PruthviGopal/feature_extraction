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
#define USE_COPY_MEMORY 1

__global__ void RemovePointKernel_init(float* img, char* keep_or_remove, int n) {

    // Calculate global thread index based on the block and thread indices ----
    int i = threadIdx.x + blockDim.x* blockIdx.x;

    if(i>=n) {
        return;
    }

    //set keep as default
    keep_or_remove[i] = 1;
}


__global__ void RemovePointKernel(float* img, int* knn_data, char* keep_or_remove, int n, int iter, float distanceThreshold, float angleThreshold) {

    // Calculate global thread index based on the block and thread indices ----
    int i = threadIdx.x + blockDim.x* blockIdx.x;
    i += iter;

    if(i>=n || iter>=n-1 || keep_or_remove[iter] == 0) {
        return;
    }

    float*p1, *p2, *p3;

    p1 = img + i * 3;
    p2 = img + iter * 3;

    float length  = pow(*p1 - *p2, 2);
    length += pow(*(p1 + 1) - *(p2 + 1), 2);
    length += pow(*(p1 + 2) - *(p2 + 2), 2);
    if(length > distanceThreshold) {
        return;
    }

    //plane ax+by+cz+d=0
    //calculate a, b, c
    //a = ( (p2.y-p1.y) * (p3.z-p1.z) - (p2.z-p1.z) * (p3.y-p1.y) );  
    //b = ( (p2.z-p1.z) * (p3.x-p1.x) - (p2.x-p1.x) * (p3.z-p1.z) );  
    //c = ( (p2.x-p1.x) * (p3.y-p1.y) - (p2.y-p1.y) * (p3.x-p1.x) );  
    //d = ( 0-(a*p1.x+b*p1.y+c*p1.z) );  
    p1 = img + i * 3;
    p2 = img + (*(knn_data + i*2)) * 3;
    p3 = img + (*(knn_data + i*2 + 1)) * 3;
    float a1 = (*(p2 + 1) - *(p1 + 1)) * (*(p3 + 2) - *(p1 + 2)) - (*(p2 + 2) - *(p1 + 2)) * (*(p3 + 1) - *(p1 + 1));
    float b1 = (*(p2 + 2) - *(p1 + 2)) * (*p3 - *p1) - (*p2 - *p1) * (*(p3 + 2) - *(p1 + 2));
    float c1 = (*p2 - *p1) * (*(p3 + 1) - *(p1 + 1)) - (*(p2 + 1) - *(p1 + 1)) * (*p3 - *p1);  

    p1 = img + iter * 3;
    p2 = img + (*(knn_data + iter*2)) * 3;
    p3 = img + (*(knn_data + iter*2 + 1)) * 3;
    float a2 = (*(p2 + 1) - *(p1 + 1)) * (*(p3 + 2) - *(p1 + 2)) - (*(p2 + 2) - *(p1 + 2)) * (*(p3 + 1) - *(p1 + 1));
    float b2 = (*(p2 + 2) - *(p1 + 2)) * (*p3 - *p1) - (*p2 - *p1) * (*(p3 + 2) - *(p1 + 2));  
    float c2 = (*p2 - *p1) * (*(p3 + 1) - *(p1 + 1)) - (*(p2 + 1) - *(p1 + 1)) * (*p3 - *p1);  

    //calculate the angle between i and j
    float cos_angle = abs((double)a1 * (double)a2 + (double)b1 * (double)b2 + (double)c1 * (double)c2) / (sqrt(pow((double)a1, 2) + pow((double)b1, 2) + pow((double)c1, 2)) * sqrt(pow((double)a2, 2) + pow((double)b2, 2) + pow((double)c2, 2)));

    //if the angle is less than threshold, set keep_or_remove = 0
    if(cos_angle >= angleThreshold) {
        keep_or_remove[i] = 0;
    }
}

bool remove_point(float* d_img, int* d_knn_data, char* keep_or_remove, int point_num, float distanceThreshold, float angleThreshold) {
    clock_t begin, end;

    //////////////////////////////////////////////////////////////
    // Allocate device variables
    //////////////////////////////////////////////////////////////
    printf("Allocating device variables of keep_or_remove...\n"); 
    fflush(stdout);
    begin = clock();

    //INSERT CODE HERE
    char* d_keep_or_remove;
    cudaMalloc((void **)&d_keep_or_remove, sizeof(char)*point_num);

    checkCudaErrors(cudaDeviceSynchronize());
    end = clock();
    printf("Elapsed: %f seconds\n\n", (double)(end - begin) / CLOCKS_PER_SEC);

    //////////////////////////////////////////////////////////////
    // Launch kernel
    //////////////////////////////////////////////////////////////
    printf("Launching kernel of remove point...\n"); 
    fflush(stdout);
    begin = clock();

    int BLOCK_SIZE = 256;

    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid(ceil(point_num / (float)BLOCK_SIZE), 1, 1);

    RemovePointKernel_init <<<dimGrid, dimBlock >>>(d_img, d_keep_or_remove, point_num);

#if USE_COPY_MEMORY > 0
    for(int i=0; i<point_num-1; i++) {
        if(*(keep_or_remove + i) == 0 && i>0) {
            continue;
        }

        int kernel_start_point = i+1;
        dim3 dimGrid(ceil((point_num - kernel_start_point) / (float)BLOCK_SIZE), 1, 1);
        RemovePointKernel <<<dimGrid, dimBlock >>>(d_img, d_knn_data, d_keep_or_remove, point_num, i, distanceThreshold, angleThreshold);

        checkCudaErrors(cudaDeviceSynchronize());
        //end = clock();
        //printf("Elapsed: %f seconds\n\n", (double)(end - begin) / CLOCKS_PER_SEC);

        //////////////////////////////////////////////////////////////
        // Copy device variables from host
        //////////////////////////////////////////////////////////////
        //printf("Copying data from device to host of keep_or_remove...\n"); 
        //fflush(stdout);
        //begin = clock();

        cudaMemcpy(keep_or_remove + kernel_start_point, d_keep_or_remove + kernel_start_point, sizeof(char)*(point_num-kernel_start_point), cudaMemcpyDeviceToHost);

        checkCudaErrors(cudaDeviceSynchronize());
        //end = clock();
        //printf("Elapsed: %f seconds\n\n", (double)(end - begin) / CLOCKS_PER_SEC);

    }

#else
    for(int i=0; i<point_num-1; i++) {
        dim3 dimGrid(ceil((point_num - i - 1) / (float)BLOCK_SIZE), 1, 1);
        RemovePointKernel <<<dimGrid, dimBlock >>>(d_img, d_knn_data, d_keep_or_remove, point_num, i, distanceThreshold, angleThreshold);
        checkCudaErrors(cudaDeviceSynchronize());
    }

#endif

    end = clock();
    printf("Elapsed: %f seconds\n\n", (double)(end - begin) / CLOCKS_PER_SEC);


#if USE_COPY_MEMORY == 0
    //////////////////////////////////////////////////////////////
    // Copy device variables from host
    //////////////////////////////////////////////////////////////
    printf("Copying data from device to host of keep_or_remove...\n"); 
    begin = clock();

    cudaMemcpy(keep_or_remove, d_keep_or_remove, sizeof(char)*point_num, cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaDeviceSynchronize());

    end = clock();
    printf("Elapsed: %f seconds\n\n", (double)(end - begin) / CLOCKS_PER_SEC);
#endif

    //////////////////////////////////////////////////////////////
    // Free memory
    //////////////////////////////////////////////////////////////
    cudaFree(d_keep_or_remove);

    return 0;
}

bool remove_point_c(float* img, knn2 *knn2_arr, int point_num, float distanceThreshold, float angleThreshold) {
    clock_t begin, end;

    //////////////////////////////////////////////////////////////
    // remove unused points
    //////////////////////////////////////////////////////////////
    printf("Running remove_point_c...\n"); 
    fflush(stdout);
    begin =clock();

    for(int i=0; i<point_num; i++) {
        (knn2_arr + i)->keep_or_remove = true;
    }

    for(int i=0; i<point_num-1; i++) {
        if((knn2_arr + i)->keep_or_remove == false && i>0) {
            continue;
        }

        float*p1, *p2, *p3;

        //plane ax+by+cz+d=0
        //calculate a, b, c
        //a = ( (p2.y-p1.y) * (p3.z-p1.z) - (p2.z-p1.z) * (p3.y-p1.y) );  
        //b = ( (p2.z-p1.z) * (p3.x-p1.x) - (p2.x-p1.x) * (p3.z-p1.z) );  
        //c = ( (p2.x-p1.x) * (p3.y-p1.y) - (p2.y-p1.y) * (p3.x-p1.x) );  
        //d = ( 0-(a*p1.x+b*p1.y+c*p1.z) );  
        p1 = img + i * 3;
        p2 = img + (knn2_arr + i)->data0 * 3;
        p3 = img + (knn2_arr + i)->data1 * 3;
        float a1 = (*(p2 + 1) - *(p1 + 1)) * (*(p3 + 2) - *(p1 + 2)) - (*(p2 + 2) - *(p1 + 2)) * (*(p3 + 1) - *(p1 + 1));
        float b1 = (*(p2 + 2) - *(p1 + 2)) * (*p3 - *p1) - (*p2 - *p1) * (*(p3 + 2) - *(p1 + 2));
        float c1 = (*p2 - *p1) * (*(p3 + 1) - *(p1 + 1)) - (*(p2 + 1) - *(p1 + 1)) * (*p3 - *p1);  

        for(int j=i+1; j<point_num; j++) {
            p1 = img + i * 3;
            p2 = img + j * 3;

            float length  = pow(*p1 - *p2, 2);
            length += pow(*(p1 + 1) - *(p2 + 1), 2);
            length += pow(*(p1 + 2) - *(p2 + 2), 2);
            if(length > distanceThreshold) {
                continue;
            }

            p1 = img + j * 3;
            p2 = img + (knn2_arr + j)->data0 * 3;
            p3 = img + (knn2_arr + j)->data1 * 3;
            float a2 = (*(p2 + 1) - *(p1 + 1)) * (*(p3 + 2) - *(p1 + 2)) - (*(p2 + 2) - *(p1 + 2)) * (*(p3 + 1) - *(p1 + 1));
            float b2 = (*(p2 + 2) - *(p1 + 2)) * (*p3 - *p1) - (*p2 - *p1) * (*(p3 + 2) - *(p1 + 2));  
            float c2 = (*p2 - *p1) * (*(p3 + 1) - *(p1 + 1)) - (*(p2 + 1) - *(p1 + 1)) * (*p3 - *p1);  

            //calculate the angle between i and j
            float cos_angle = abs((double)a1 * (double)a2 + (double)b1 * (double)b2 + (double)c1 * (double)c2) / (sqrt(pow((double)a1, 2) + pow((double)b1, 2) + pow((double)c1, 2)) * sqrt(pow((double)a2, 2) + pow((double)b2, 2) + pow((double)c2, 2)));

            //if the angle is less than threshold, set keep_or_remove = 0
            if(cos_angle >= angleThreshold) {
                (knn2_arr + i)->keep_or_remove = false;
            }

        }

    }

    end = clock();
    printf("Elapsed: %f seconds\n\n", (double)(end - begin) / CLOCKS_PER_SEC);

    return 0;
}

