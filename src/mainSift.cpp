//********************************************************//
// CUDA SIFT extractor by Marten Björkman aka Celebrandil //
//              celle @ csc.kth.se                       //
//********************************************************//

#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "cudaImage.h"
#include "cudaSift.h"
#include "cudautils.h"

int ImproveHomography(SiftData &data, float *homography, int numLoops,
                      float minScore, float maxAmbiguity, float thresh);
void PrintMatchData(SiftData &siftData1, SiftData &siftData2, CudaImage &img);
void MatchAll(SiftData &siftData1, SiftData &siftData2, float *homography);
void PlotKeyPoints(SiftData &siftData1, CudaImage &img);

void siftRunImage(int devnum, int imgSet);
void siftRunImageUnified(int devNum, int imgSet);
void siftRunVideo(int devnum, int imgSet);
void siftRunVideoAsync(int devNum, int imgSet);
void siftRunVideoUnifiedSync(int devnum, int imgSet);
void siftRunVideoUnifiedAsync(int devNum, int imgSet);
void siftRunVideoCPU(int devNum, int imgSet);
double ScaleUp(CudaImage &res, CudaImage &src);

#define WIDTH 3840
#define HEIGHT 2160
#define N 100
#define WARMUP 5
int LIVE = 0;

enum {
    IMAGE,
    IMAGE_UNIFIED,
    VIDEO_SYNC,
    VIDEO_ASYNC,
    VIDEO_UNIFIED_SYNC,
    VIDEO_UNIFIED_ASYNC,
    CPU
};

char *fname = ((char *)"data/vid1.mp4");

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
main(int argc, char **argv) {
    int devNum = 0;
    int imgSet = 0;
    if (argc > 1) imgSet = std::atoi(argv[1]);
    if (argc > 2) LIVE = std::atoi(argv[2]);
    if (argc > 3) fname = argv[3];
    printf("LIVE : %d ", LIVE);
    switch (imgSet) {
        case VIDEO_UNIFIED_SYNC:
            printf("Running SYNCHRONOUS SIFT with UNIFIED MEMORY on video\n");
            siftRunVideoUnifiedSync(devNum, imgSet);
            break;
        case VIDEO_SYNC:
            printf("Running SYNCHRONOUS SIFT on video\n");
            siftRunVideo(devNum, imgSet);
            break;
        case IMAGE:
            printf("Running Naive SIFT on image\n");
            siftRunImage(devNum, imgSet);
            break;
        case IMAGE_UNIFIED:
            printf("Running SIFT on image with unified memory\n");
            siftRunImageUnified(devNum, imgSet);
            break;
        case VIDEO_UNIFIED_ASYNC:
            printf("Running ASYNCHRONOUS SIFT with UNIFIED MEMORY on video\n");
            siftRunVideoUnifiedAsync(devNum, imgSet);
            break;
        case VIDEO_ASYNC:
            printf("Running ASYNCHRONOUS SIFT on video\n");
            siftRunVideoAsync(devNum, imgSet);
            break;
        case CPU:
            printf("Running CPU SIFT on video\n");
            siftRunVideoCPU(devNum, imgSet);
            break;

        default:
            printf("Not implemented\n");
    }
    return 0;
}
void siftRunImageUnified(int devNum, int imgSet) {
    void *unified_ptr;
    cudaMallocManaged(&unified_ptr, WIDTH * HEIGHT * sizeof(float));
    // Read images using OpenCV
    cv::Mat curr_img(HEIGHT, WIDTH, CV_32FC1, unified_ptr);
    cv::Mat rimg, grey;
    cv::VideoCapture cap;
    if (!LIVE)
        cap.open(fname);
    else
        cap.open(0);
    if (!cap.isOpened()) {
        printf("Error opening file\n");
        return;
    }
    cap.read(rimg);
    cv::cvtColor(rimg, grey, cv::COLOR_BGR2GRAY);
    grey.convertTo(curr_img, CV_32FC1);

    unsigned int w = curr_img.cols;
    unsigned int h = curr_img.rows;
    std::cout << "Image size = (" << w << "," << h << ")" << std::endl;

    // Initial Cuda images and download images to device
    std::cout << "Initializing data..." << std::endl;
    InitCuda(devNum);
    CudaImage img1;
    img1.AllocateUnified(w, h, w, false, NULL, (float *)curr_img.data);

    // Extract Sift features from images
    SiftData siftData1;
    float initBlur = 1.0f;
    float thresh = 4.5f;
    InitSiftData(siftData1, 32768, true, true);

    float *memoryTmp = AllocSiftTempMemory(w, h, 5, false);
    double average_time = 0;
    for (int i = 0; i < N; i++) {
        double temp_time = ExtractSift(siftData1, img1, 5, initBlur, thresh,
                                       0.0f, false, memoryTmp);
        if (i >= WARMUP) average_time += temp_time;
    }
    FreeSiftTempMemory(memoryTmp);
    printf("Average Time per run: %.4f\n", average_time / (N - WARMUP));
    PlotKeyPoints(siftData1, img1);
    cv::imwrite("data/curr_img_pts.pgm", curr_img);

    // Free Sift data from device
    FreeSiftData(siftData1);
}
void siftRunImage(int devNum, int imgSet) {
    cv::VideoCapture cap;
    if (LIVE)
        cap.open(0);
    else
        cap.open(fname);
    if (!cap.isOpened()) {
        printf("Error opening file\n");
        return;
    }
    cv::Mat curr_img, grey;
    cv::Mat rimg;
    cap.read(rimg);
    cv::cvtColor(rimg, grey, cv::COLOR_BGR2GRAY);
    grey.convertTo(curr_img, CV_32FC1);

    unsigned int w = curr_img.cols;
    unsigned int h = curr_img.rows;
    std::cout << "Image size = (" << w << "," << h << ")" << std::endl;

    // Initial Cuda images and download images to device
    std::cout << "Initializing data..." << std::endl;
    InitCuda(devNum);
    CudaImage img1;
    img1.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float *)curr_img.data);
    img1.Download();

    // Extract Sift features from images
    SiftData siftData1;
    float initBlur = 1.0f;
    float thresh = 4.5f;
    InitSiftData(siftData1, 32768, true, true);

    float *memoryTmp = AllocSiftTempMemory(w, h, 5, false);
    double average_time = 0;
    for (int i = 0; i < N; i++) {
        double temp_time = ExtractSift(siftData1, img1, 5, initBlur, thresh,
                                       0.0f, false, memoryTmp);
        if (i >= WARMUP) average_time += temp_time;
    }
    FreeSiftTempMemory(memoryTmp);
    printf("Average Time per run: %.4f\n", average_time / (N - WARMUP));
    PlotKeyPoints(siftData1, img1);
    cv::imwrite("data/curr_img_pts.pgm", curr_img);

    // Free Sift data from device
    FreeSiftData(siftData1);
}
void siftRunVideoCPU(int devNum, int imgSet) {
    // Read images using OpenCV
    cv::Mat curr_img;
    cv::Mat img_keypoints;
    cv::VideoCapture cap;
    if (LIVE)
        cap.open(0);
    else
        cap.open(fname);
    if (!cap.isOpened()) {
        printf("Error opening file\n");
        return;
    }

    float initBlur = 1.0f;
    float thresh = 4.5f;
    unsigned int w = curr_img.cols;
    unsigned int h = curr_img.rows;

    double average_time = 0;
    double average_extraction_time = 0;

    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    std::vector<cv::KeyPoint> keypoints;

    int i;
    for (i = 0;; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        cap.read(curr_img);
        if (curr_img.empty()) {
            break;
        }

        detector->detect(curr_img, keypoints);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff = end - start;
        double temp_time = diff.count();
        if (i >= WARMUP) average_extraction_time += temp_time;

        cv::drawKeypoints(curr_img, keypoints, img_keypoints);

        end = std::chrono::high_resolution_clock::now();
        diff = end - start;
        if (i >= WARMUP) average_time += diff.count();

        cv::imshow("Video", img_keypoints);
        if ((char)cv::waitKey(1) == 27) break;
    }
    printf("Average Time per frame: %.4f\n", average_time / (i - WARMUP));
    printf("Average Time per extraction: %.4f\n",
           average_extraction_time / (i - WARMUP));
};
void siftRunVideo(int devNum, int imgSet) {
    // Read images using OpenCV
    cv::Mat curr_img(HEIGHT, WIDTH, CV_32FC1);
    cv::Mat rimg, grey;
    cv::VideoCapture cap;
    if (LIVE)
        cap.open(0);
    else
        cap.open(fname);
    if (!cap.isOpened()) {
        printf("Error opening file\n");
        return;
    }

    InitCuda(devNum);
    SiftData siftData1;
    float initBlur = 1.0f;
    float thresh = 4.5f;
    InitSiftData(siftData1, 32768, true, true);
    unsigned int w = curr_img.cols;
    unsigned int h = curr_img.rows;
    CudaImage img1;
    img1.Allocate(w, h, w, false, NULL, (float *)curr_img.data);

    float *memoryTmp = AllocSiftTempMemory(w, h, 5, false);
    double average_time = 0;
    double average_extraction_time = 0;

    int i;
    for (i = 0;; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        cap.read(rimg);
        if (rimg.empty()) {
            break;
        }
        cv::cvtColor(rimg, grey, cv::COLOR_BGR2GRAY);
        grey.convertTo(curr_img, CV_32FC1);

        img1.Download();
        double temp_time = ExtractSift(siftData1, img1, 5, initBlur, thresh,
                                       0.0f, false, memoryTmp);
        if (i >= WARMUP) average_extraction_time += temp_time;
        PlotKeyPoints(siftData1, img1);
        auto end = std::chrono::high_resolution_clock::now();
        curr_img.convertTo(rimg, CV_32FC1, 1.0 / 255);
        std::chrono::duration<double, std::milli> diff = end - start;
        if (i >= WARMUP) average_time += diff.count();
        cv::imshow("Video", rimg);
        if ((char)cv::waitKey(1) == 27) break;
    }
    FreeSiftTempMemory(memoryTmp);
    printf("Average Time per frame: %.4f\n", average_time / (i - WARMUP));
    printf("Average Time per extraction: %.4f\n",
           average_extraction_time / (i - WARMUP));
    // Free Sift data from device
    FreeSiftData(siftData1);
};
void siftRunVideoAsync(int devNum, int imgSet) {
    // Unified ptr images
    cv::Mat curr_img(HEIGHT, WIDTH, CV_32FC1);
    cv::Mat next_img(HEIGHT, WIDTH, CV_32FC1);

    // temp images to read using openCV
    cv::Mat rimg, grey, show;
    cv::VideoCapture cap;
    if (LIVE)
        cap.open(0);
    else
        cap.open(fname);
    if (!cap.isOpened()) {
        printf("Error opening file\n");
        return;
    }

    InitCuda(devNum);
    SiftData siftData1, siftData2;
    float initBlur = 1.0f;
    float thresh = 4.5f;
    InitSiftData(siftData1, 32768, true, true);
    InitSiftData(siftData2, 32768, true, true);
    unsigned int w = curr_img.cols;
    unsigned int h = curr_img.rows;
    CudaImage img1, img2;
    img1.Allocate(w, h, w, false, NULL, (float *)curr_img.data);
    img2.Allocate(w, h, w, false, NULL, (float *)next_img.data);

    float *memoryTmp = AllocSiftTempMemory(w, h, 5, false);
    double average_time = 0;
    double average_extraction_time = 0;
    cudaStream_t st1, st2;
    safeCall(cudaStreamCreate(&st1));
    safeCall(cudaStreamCreate(&st2));
    int i;

    // pointers
    cv::Mat *current, *next;
    CudaImage *curr_cuda, *next_cuda;
    cudaStream_t *curr_stream, *next_stream;
    SiftData *curr_sift, *next_sift;

    current = &curr_img;
    next = &next_img;

    curr_cuda = &img1;
    next_cuda = &img2;

    curr_stream = &st1;
    next_stream = &st2;

    curr_sift = &siftData1;
    next_sift = &siftData2;

    cap.read(rimg);
    if (rimg.empty()) {
        goto cleanup;
    }

    cv::cvtColor(rimg, grey, cv::COLOR_BGR2GRAY);
    grey.convertTo(*current, CV_32FC1);
    curr_cuda->Download();

    for (i = 0;; i++) {
        auto start = std::chrono::high_resolution_clock::now();

#pragma omp sections
        {
#pragma omp section
            {
                double temp_time = ExtractSiftAsync(
                    *curr_sift, *curr_cuda, 5, initBlur, thresh, 0.0f, false,
                    memoryTmp, *curr_stream);

                if (i >= WARMUP) average_extraction_time += temp_time;
#ifdef MANAGEDMEM
                cudaStreamSynchronize(*curr_stream);
#else
                if (curr_sift->h_data)
                    safeCall(
                        cudaMemcpyAsync(curr_sift->h_data, curr_sift->d_data,
                                        sizeof(SiftPoint) * curr_sift->numPts,
                                        cudaMemcpyDeviceToHost, curr_stream));
#endif
            }
#pragma omp section
            {
                cap.read(rimg);
                if (rimg.empty()) {
                    printf("Done: %d \n", i);
                } else {
                    cv::cvtColor(rimg, grey, cv::COLOR_BGR2GRAY);
                    grey.convertTo(*next, CV_32FC1);
                    next_cuda->DownloadAsync(*next_stream);
                }
                cudaStreamSynchronize(*next_stream);
            }
        }

        if (rimg.empty()) {
            break;
        }
        // synchronized
        PlotKeyPoints(*curr_sift, *curr_cuda);
        auto end = std::chrono::high_resolution_clock::now();
        current->convertTo(show, CV_32FC1, 1.0 / 255);
        cv::Mat *tmp;
        tmp = current;
        current = next;
        next = tmp;

        CudaImage *tmp_cuda;
        tmp_cuda = curr_cuda;
        curr_cuda = next_cuda;
        next_cuda = tmp_cuda;

        cudaStream_t *tmp_stream;
        tmp_stream = curr_stream;
        curr_stream = next_stream;
        next_stream = tmp_stream;

        SiftData *tmp_sift;
        tmp_sift = curr_sift;
        curr_sift = next_sift;
        next_sift = tmp_sift;

        std::chrono::duration<double, std::milli> diff = end - start;
        if (i >= WARMUP) average_time += diff.count();
        cv::imshow("Video", show);
        if ((char)cv::waitKey(1) == 27) break;
    }

    printf("Average Time per frame: %.4f\n", average_time / (i - WARMUP));
    printf("Average Time per extraction: %.4f\n",
           average_extraction_time / (i - WARMUP));
cleanup:
    FreeSiftTempMemory(memoryTmp);
    // Free Sift data from device
    FreeSiftData(siftData1);
    FreeSiftData(siftData2);
};
void siftRunVideoUnifiedAsync(int devNum, int imgSet) {
    void *unified_ptr1, *unified_ptr2;
    cudaMallocManaged(&unified_ptr1, WIDTH * HEIGHT * sizeof(float));
    cudaMallocManaged(&unified_ptr2, WIDTH * HEIGHT * sizeof(float));

    // Unified ptr images
    cv::Mat curr_img(HEIGHT, WIDTH, CV_32FC1, unified_ptr1);
    cv::Mat next_img(HEIGHT, WIDTH, CV_32FC1, unified_ptr2);

    // temp images to read using openCV
    cv::Mat rimg, grey, show;
    cv::VideoCapture cap;
    if (LIVE)
        cap.open(0);
    else
        cap.open(fname);
    if (!cap.isOpened()) {
        printf("Error opening file\n");
        return;
    }

    InitCuda(devNum);
    SiftData siftData1, siftData2;
    float initBlur = 1.0f;
    float thresh = 4.5f;
    InitSiftData(siftData1, 32768, true, true);
    InitSiftData(siftData2, 32768, true, true);
    unsigned int w = curr_img.cols;
    unsigned int h = curr_img.rows;
    CudaImage img1, img2;
    img1.AllocateUnified(w, h, w, false, NULL, (float *)curr_img.data);
    img2.AllocateUnified(w, h, w, false, NULL, (float *)next_img.data);

    float *memoryTmp = AllocSiftTempMemory(w, h, 5, false);
    double average_time = 0;
    double average_extraction_time = 0;
    cudaStream_t st1, st2;
    cudaStreamCreate(&st1);
    cudaStreamCreate(&st2);
    int i;

    // pointers
    cv::Mat *current, *next;
    CudaImage *curr_cuda, *next_cuda;
    cudaStream_t *curr_stream, *next_stream;
    SiftData *curr_sift, *next_sift;

    current = &curr_img;
    next = &next_img;

    curr_cuda = &img1;
    next_cuda = &img2;

    curr_stream = &st1;
    next_stream = &st2;

    curr_sift = &siftData1;
    next_sift = &siftData2;

    cap.read(rimg);
    if (rimg.empty()) {
        goto cleanup;
    }

    cv::cvtColor(rimg, grey, cv::COLOR_BGR2GRAY);
    grey.convertTo(*current, CV_32FC1);

    for (i = 0;; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        double temp_time =
            ExtractSiftAsync(*curr_sift, *curr_cuda, 5, initBlur, thresh, 0.0f,
                             false, memoryTmp, *curr_stream);

        if (i >= WARMUP) average_extraction_time += temp_time;

        cap.read(rimg);
        if (rimg.empty()) {
            printf("Done: %d \n", i);
            break;
        }
        cv::cvtColor(rimg, grey, cv::COLOR_BGR2GRAY);
        grey.convertTo(*next, CV_32FC1);
#ifdef MANAGEDMEM
        cudaStreamSynchronize(*curr_stream);
#else
        if (curr_sift->h_data)
            safeCall(cudaMemcpyAsync(curr_sift->h_data, curr_sift->d_data,
                                     sizeof(SiftPoint) * curr_sift->numPts,
                                     cudaMemcpyDeviceToHost, *curr_stream));
#endif
        // cudaStreamSynchronize(*next_stream);
        // synchronized
        PlotKeyPoints(*curr_sift, *curr_cuda);
        auto end = std::chrono::high_resolution_clock::now();
        current->convertTo(show, CV_32FC1, 1.0 / 255);
        cv::Mat *tmp;
        tmp = current;
        current = next;
        next = tmp;

        CudaImage *tmp_cuda;
        tmp_cuda = curr_cuda;
        curr_cuda = next_cuda;
        next_cuda = tmp_cuda;

        cudaStream_t *tmp_stream;
        tmp_stream = curr_stream;
        curr_stream = next_stream;
        next_stream = tmp_stream;

        SiftData *tmp_sift;
        tmp_sift = curr_sift;
        curr_sift = next_sift;
        next_sift = tmp_sift;

        std::chrono::duration<double, std::milli> diff = end - start;
        if (i >= WARMUP) average_time += diff.count();
        cv::imshow("Video", show);
        if ((char)cv::waitKey(1) == 27) break;
    }

    printf("Average Time per frame: %.4f\n", average_time / (i - WARMUP));
    printf("Average Time per extraction: %.4f\n",
           average_extraction_time / (i - WARMUP));
cleanup:
    FreeSiftTempMemory(memoryTmp);
    // Free Sift data from device
    FreeSiftData(siftData1);
    FreeSiftData(siftData2);
};
void siftRunVideoUnifiedSync(int devNum, int imgSet) {
    void *unified_ptr;
    cudaMallocManaged(&unified_ptr, WIDTH * HEIGHT * sizeof(float));
    // Read images using OpenCV
    cv::Mat curr_img(HEIGHT, WIDTH, CV_32FC1, unified_ptr);
    cv::Mat rimg, grey;
    cv::VideoCapture cap;
    if (!LIVE)
        cap.open(fname);
    else
        cap.open(0);
    if (!cap.isOpened()) {
        printf("Error opening file\n");
        return;
    }

    InitCuda(devNum);
    SiftData siftData1;
    float initBlur = 1.0f;
    float thresh = 4.5f;
    InitSiftData(siftData1, 32768, true, true);
    unsigned int w = curr_img.cols;
    unsigned int h = curr_img.rows;
    CudaImage img1;
    img1.AllocateUnified(w, h, w, false, NULL, (float *)curr_img.data);

    float *memoryTmp = AllocSiftTempMemory(w, h, 5, false);
    double average_time = 0;
    double average_extraction_time = 0;

    int i;
    for (i = 0;; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        cap.read(rimg);
        if (rimg.empty()) {
            break;
        }

        cv::cvtColor(rimg, grey, cv::COLOR_BGR2GRAY);
        grey.convertTo(curr_img, CV_32FC1);

        double temp_time = ExtractSift(siftData1, img1, 5, initBlur, thresh,
                                       0.0f, false, memoryTmp);
        if (i >= WARMUP) average_extraction_time += temp_time;
        PlotKeyPoints(siftData1, img1);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff = end - start;
        if (i >= WARMUP) average_time += diff.count();
        curr_img.convertTo(rimg, CV_32FC1, 1.0 / 255);
        cv::imshow("Video", rimg);
        if ((char)cv::waitKey(1) == 27) break;
    }
    FreeSiftTempMemory(memoryTmp);
    printf("Average Time per frame: %.4f\n", average_time / (i - WARMUP));
    printf("Average Time per extraction: %.4f\n",
           average_extraction_time / (i - WARMUP));
    // Free Sift data from device
    FreeSiftData(siftData1);
};

void MatchAll(SiftData &siftData1, SiftData &siftData2, float *homography) {
#ifdef MANAGEDMEM
    SiftPoint *sift1 = siftData1.m_data;
    SiftPoint *sift2 = siftData2.m_data;
#else
    SiftPoint *sift1 = siftData1.h_data;
    SiftPoint *sift2 = siftData2.h_data;
#endif
    int numPts1 = siftData1.numPts;
    int numPts2 = siftData2.numPts;
    int numFound = 0;
#if 1
    homography[0] = homography[4] = -1.0f;
    homography[1] = homography[3] = homography[6] = homography[7] = 0.0f;
    homography[2] = 1279.0f;
    homography[5] = 959.0f;
#endif
    for (int i = 0; i < numPts1; i++) {
        float *data1 = sift1[i].data;
        std::cout << i << ":" << sift1[i].scale << ":"
                  << (int)sift1[i].orientation << " " << sift1[i].xpos << " "
                  << sift1[i].ypos << std::endl;
        bool found = false;
        for (int j = 0; j < numPts2; j++) {
            float *data2 = sift2[j].data;
            float sum = 0.0f;
            for (int k = 0; k < 128; k++) sum += data1[k] * data2[k];
            float den = homography[6] * sift1[i].xpos +
                        homography[7] * sift1[i].ypos + homography[8];
            float dx = (homography[0] * sift1[i].xpos +
                        homography[1] * sift1[i].ypos + homography[2]) /
                           den -
                       sift2[j].xpos;
            float dy = (homography[3] * sift1[i].xpos +
                        homography[4] * sift1[i].ypos + homography[5]) /
                           den -
                       sift2[j].ypos;
            float err = dx * dx + dy * dy;
            if (err < 100.0f)  // 100.0
                found = true;
            if (err < 100.0f || j == sift1[i].match) {  // 100.0
                if (j == sift1[i].match && err < 100.0f)
                    std::cout << " *";
                else if (j == sift1[i].match)
                    std::cout << " -";
                else if (err < 100.0f)
                    std::cout << " +";
                else
                    std::cout << "  ";
                std::cout << j << ":" << sum << ":" << (int)sqrt(err) << ":"
                          << sift2[j].scale << ":" << (int)sift2[j].orientation
                          << " " << sift2[j].xpos << " " << sift2[j].ypos << " "
                          << (int)dx << " " << (int)dy << std::endl;
            }
        }
        std::cout << std::endl;
        if (found) numFound++;
    }
    std::cout << "Number of finds: " << numFound << " / " << numPts1
              << std::endl;
    std::cout << homography[0] << " " << homography[1] << " " << homography[2]
              << std::endl;  //%%%
    std::cout << homography[3] << " " << homography[4] << " " << homography[5]
              << std::endl;  //%%%
    std::cout << homography[6] << " " << homography[7] << " " << homography[8]
              << std::endl;  //%%%
}

void PlotKeyPoints(SiftData &siftData1, CudaImage &img) {
    int numPts = siftData1.numPts;
#ifdef MANAGEDMEM
    SiftPoint *sift1 = siftData1.m_data;
#else
    SiftPoint *sift1 = siftData1.h_data;
#endif
    float *h_img = img.h_data;
    int w = img.width;
    int h = img.height;
    std::cout << std::setprecision(3);

#pragma omp parallel for
    for (int j = 0; j < numPts; j++) {
        int k = sift1[j].match;
        int x = (int)(sift1[j].xpos + 0.5);
        int y = (int)(sift1[j].ypos + 0.5);
        int s = std::min(
            x, std::min(y, std::min(w - x - 2,
                                    std::min(h - y - 2,
                                             (int)(1.41 * sift1[j].scale)))));
        int p = y * w + x;
        p += (w + 1);
#pragma omp parallel for
        for (int k = 0; k < s; k++)
            h_img[p - k] = h_img[p + k] = h_img[p - k * w] = h_img[p + k * w] =
                0.0f;
        p -= (w + 1);
#pragma omp parallel for
        for (int k = 0; k < s; k++)
            h_img[p - k] = h_img[p + k] = h_img[p - k * w] = h_img[p + k * w] =
                255.0f;
    }
    std::cout << std::setprecision(6);
}

void PrintMatchData(SiftData &siftData1, SiftData &siftData2, CudaImage &img) {
    int numPts = siftData1.numPts;
#ifdef MANAGEDMEM
    SiftPoint *sift1 = siftData1.m_data;
    SiftPoint *sift2 = siftData2.m_data;
#else
    SiftPoint *sift1 = siftData1.h_data;
    SiftPoint *sift2 = siftData2.h_data;
#endif
    float *h_img = img.h_data;
    int w = img.width;
    int h = img.height;
    std::cout << std::setprecision(3);
    for (int j = 0; j < numPts; j++) {
        int k = sift1[j].match;
        if (sift1[j].match_error < 5) {
            float dx = sift2[k].xpos - sift1[j].xpos;
            float dy = sift2[k].ypos - sift1[j].ypos;
#if 0
      if (false && sift1[j].xpos>550 && sift1[j].xpos<600) {
	std::cout << "pos1=(" << (int)sift1[j].xpos << "," << (int)sift1[j].ypos << ") ";
	std::cout << j << ": " << "score=" << sift1[j].score << "  ambiguity=" << sift1[j].ambiguity << "  match=" << k << "  ";
	std::cout << "scale=" << sift1[j].scale << "  ";
	std::cout << "error=" << (int)sift1[j].match_error << "  ";
	std::cout << "orient=" << (int)sift1[j].orientation << "," << (int)sift2[k].orientation << "  ";
	std::cout << " delta=(" << (int)dx << "," << (int)dy << ")" << std::endl;
      }
#endif
#if 1
            int len = (int)(fabs(dx) > fabs(dy) ? fabs(dx) : fabs(dy));
            for (int l = 0; l < len; l++) {
                int x = (int)(sift1[j].xpos + dx * l / len);
                int y = (int)(sift1[j].ypos + dy * l / len);
                h_img[y * w + x] = 255.0f;
            }
#endif
        }
        int x = (int)(sift1[j].xpos + 0.5);
        int y = (int)(sift1[j].ypos + 0.5);
        int s = std::min(
            x, std::min(y, std::min(w - x - 2,
                                    std::min(h - y - 2,
                                             (int)(1.41 * sift1[j].scale)))));
        int p = y * w + x;
        p += (w + 1);
        for (int k = 0; k < s; k++)
            h_img[p - k] = h_img[p + k] = h_img[p - k * w] = h_img[p + k * w] =
                0.0f;
        p -= (w + 1);
        for (int k = 0; k < s; k++)
            h_img[p - k] = h_img[p + k] = h_img[p - k * w] = h_img[p + k * w] =
                255.0f;
    }
    std::cout << std::setprecision(6);
}

