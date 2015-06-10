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

#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>

#include "knn2_struct.h"

using std::string;
using std::vector;

using std::cout;
using std::cerr;
using std::endl;

extern bool knn(float*, int*, int);
extern bool knn_c(float*, knn2*, int);
extern bool remove_point(float*, int*, char*, int, float, float);
extern bool remove_point_c(float*, knn2*, int, float, float);

int main(int argc, char**argv) {
    //////////////////////////////////////////////////////////////
    // Initialize host variables
    //////////////////////////////////////////////////////////////
    float distanceThreshold = 1;
    float angleThreshold = 1-1e-5;
    bool argvIsValue = false;
    string datasetFile = "dataset";

    if(argc>1) {
        for(int i=1; i<argc; i++) {
            if(argvIsValue == true) {
                argvIsValue = false;
                continue;
            }

            if(strcmp(argv[i], "-d") == 0) {
                argvIsValue = true;
                distanceThreshold = atof(argv[i+1]);
            } else if(strcmp(argv[i], "-a") == 0) {
                argvIsValue = true;
                angleThreshold = atof(argv[i+1]);
            } else if(strcmp(argv[i], "-f") == 0) {
                argvIsValue = true;
                datasetFile = argv[i+1];
            } else {
                cerr << "Usage: " << argv[0] << " [-d distance] [-a angle] [-f dataset]" << endl;
                return 1;
            }
        }
    }

    cout << "distrance threshold: " << distanceThreshold << endl;
    cout << "angle threshold: " << angleThreshold << endl;
    cout << "dataset file: " << datasetFile << endl;

    //////////////////////////////////////////////////////////////
    // read dataset
    //////////////////////////////////////////////////////////////
    std::ifstream ifs;
    std::ofstream ofs;
    clock_t begin, end;

    ifs.open(datasetFile);
    if(!ifs) {
        cerr << "Error: file " << datasetFile << " cannot be opened!" << endl;
        return 1;
    }

    vector<float> dataset;
    std::string field;
    while(!ifs.eof()) {
        getline(ifs, field, '\n');
        std::istringstream is(field);
        string word;
        while(getline(is, word, ',')) {
            float f; 
            std::istringstream(word) >> f;
            dataset.push_back(f);
        }
    }

#ifdef DEBUG
    cout << dataset.size() << endl;
#endif
    int dataset_size = dataset.size();
    if(dataset_size % 3 != 0) {
        cerr << "Error: Image dataset should be multiple of 3" << endl;
        return 1;
    }
    int point_num = dataset_size / 3;
    float* img_h = &dataset[0];
    //float* img_h = new float[dataset.size()];
    //std::copy(dataset.begin(), dataset.end(), img_h);

#ifdef DEBUG
    //////////////////////////////////////////////////////////////
    // write dataset to check
    // check if the read data is correct by writing the data to a file
    //////////////////////////////////////////////////////////////
    //std::ofstream ofs ("dataset.txt");
    ofs.open("dataset.txt");
    if (ofs.is_open()) {
        int i=0;
        for(vector<float>::iterator it=dataset.begin(); it!=dataset.end(); ++it) {
            ofs << *it << "\t";
            i++;
            if(i==3) {
                ofs << "\n";
                i=0;
            }
        }
        ofs.close();
    }
#endif

    //////////////////////////////////////////////////////////////
    //Allocate device variables
    //////////////////////////////////////////////////////////////
    printf("Allocating device variables for d_img and d_knn_data...\n"); 
    fflush(stdout);
    begin = clock();

    float *d_img;
    int *d_knn_data;
    cudaMalloc((void **)&d_img, sizeof(float)*point_num*3);
    cudaMalloc((void **)&d_knn_data, sizeof(int)*point_num*2);

    checkCudaErrors(cudaDeviceSynchronize());
    end = clock();
    printf("Elapsed: %f seconds\n\n", (double)(end - begin) / CLOCKS_PER_SEC);

    //////////////////////////////////////////////////////////////
    // Copy host variables to device
    //////////////////////////////////////////////////////////////
    printf("Copying data from host to device for img...\n"); 
    fflush(stdout);
    begin = clock();

    cudaMemcpy(d_img, img_h, sizeof(float)*point_num*3, cudaMemcpyHostToDevice);

    checkCudaErrors(cudaDeviceSynchronize());
    end = clock();
    printf("Elapsed: %f seconds\n\n", (double)(end - begin) / CLOCKS_PER_SEC);

    //////////////////////////////////////////////////////////////
    //call knn
    //////////////////////////////////////////////////////////////
    knn(d_img, d_knn_data, point_num);

    //////////////////////////////////////////////////////////////
    // Copy device variables from host
    //////////////////////////////////////////////////////////////
    int* knn_data = new int[point_num * 2];

    printf("Copying data from device to host for knn_data...\n"); 
    fflush(stdout);
    begin = clock();

    cudaMemcpy(knn_data, d_knn_data, sizeof(int)*point_num*2, cudaMemcpyDeviceToHost);

    checkCudaErrors(cudaDeviceSynchronize());
    end = clock();
    printf("Elapsed: %f seconds\n\n", (double)(end - begin) / CLOCKS_PER_SEC);

#ifdef DEBUG
    ofs.open("knn.txt");
    if (ofs.is_open()) {
        for(int i=0; i<point_num; i++) {
            ofs << *(knn_data + i*2) << " " << *(knn_data + i*2 + 1) << endl;
        }
        ofs.close();
    }
#endif

    //////////////////////////////////////////////////////////////
    //call remove_point
    //////////////////////////////////////////////////////////////
    char* keep_or_remove = new char[point_num];
    remove_point(d_img, d_knn_data, keep_or_remove, point_num, distanceThreshold, angleThreshold);

#ifdef DEBUG
    ofs.open("removed_points.txt");
    if (ofs.is_open()) {
        for(int i=0; i<point_num; i++) {
            ofs << int(*(keep_or_remove + i)) << endl;
        }
        ofs.close();
    }
#endif

    //////////////////////////////////////////////////////////////
    // Write data set to a file
    //////////////////////////////////////////////////////////////
    string gpuDatasetFile = datasetFile + "_gpu.txt";
    ofs.open(gpuDatasetFile);
    if (ofs.is_open()) {
        for(int i=0; i<point_num; i++) {
            if(*(keep_or_remove + i) == 1) {
                ofs << *(img_h + i*3) << "," << *(img_h + i*3 + 1) << "," << *(img_h + i*3 + 2) << endl;
            }
        }
        ofs.close();
    }

#ifdef DEBUG
    ofs.open("knn_gpu");
    if (ofs.is_open()) {
        for(int i=0; i<point_num; i++) {
            ofs << *(knn_data + i*2) << ",";
            ofs << *(knn_data + i*2 + 1) << ",";
            ofs << *(keep_or_remove + i) << endl;
        }
        ofs.close();
    }
#endif

    delete knn_data;
    delete keep_or_remove;

    knn2 *knn2_arr = new knn2[point_num];

    //////////////////////////////////////////////////////////////
    // Use cpu to calculate knn
    //////////////////////////////////////////////////////////////
    //calc knn in cpu
    knn_c(img_h, knn2_arr, point_num);

#ifdef DEBUG
    ofs.open("knn_cpu");
    if (ofs.is_open()) {
        for(int i=0; i<point_num; i++) {
            ofs << (knn2_arr + i)->data0 << ",";
            ofs << (knn2_arr + i)->data1  << ",";
            ofs << (knn2_arr + i)->keep_or_remove << endl;
        }
        ofs.close();
    }
#endif
    
    //////////////////////////////////////////////////////////////
    // Use cpu to remove unused points
    //////////////////////////////////////////////////////////////
    //remove points in cpu
    remove_point_c(img_h, knn2_arr, point_num, distanceThreshold, angleThreshold);

    //////////////////////////////////////////////////////////////
    // Write data set to a file
    //////////////////////////////////////////////////////////////
    string cpuDatasetFile = datasetFile + "_c.txt";
    ofs.open(cpuDatasetFile);
    if (ofs.is_open()) {
        for(int i=0; i<point_num; i++) {
            if((knn2_arr + i)->keep_or_remove == true) {
                ofs << *(img_h + i*3) << "," << *(img_h + i*3 + 1) << "," << *(img_h + i*3 + 2) << endl;
            }
        }
        ofs.close();
    }
    
    return 0;
}

