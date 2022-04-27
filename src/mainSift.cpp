//********************************************************//
// CUDA SIFT extractor by Marten Björkman aka Celebrandil //
//              celle @ csc.kth.se                       //
//********************************************************//

#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "cudaImage.h"
#include "cudaSift.h"

int ImproveHomography(SiftData &data, float *homography, int numLoops,
                      float minScore, float maxAmbiguity, float thresh);
void PrintMatchData(SiftData &siftData1, SiftData &siftData2, CudaImage &img);
void MatchAll(SiftData &siftData1, SiftData &siftData2, float *homography);
void PlotKeyPoints(SiftData &siftData1, CudaImage &img);

double ScaleUp(CudaImage &res, CudaImage &src);

#define WIDTH 1920
#define HEIGHT 1080

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    int devNum = 0, imgSet = 0;
    if (argc > 1) devNum = std::atoi(argv[1]);
    if (argc > 2) imgSet = std::atoi(argv[2]);

    void *unified_ptr;
    cudaMallocManaged(&unified_ptr, WIDTH * HEIGHT * sizeof(float));
    // Read images using OpenCV
    cv::Mat limg(HEIGHT, WIDTH, CV_32FC1, unified_ptr);
    cv::Mat rimg;
    if (imgSet) {
        cv::imread("data/left.pgm", 0).convertTo(limg, CV_32FC1);
        cv::imread("data/righ.pgm", 0).convertTo(rimg, CV_32FC1);
    } else {
        cv::imread("data/1920x1080.png", 0).convertTo(limg, CV_32FC1);
        //     cv::imread("data/img2.png", 0).convertTo(rimg, CV_32FC1);
    }
    // cv::flip(limg, rimg, -1);
    unsigned int w = limg.cols;
    unsigned int h = limg.rows;
    std::cout << "Image size = (" << w << "," << h << ")" << std::endl;

    // Initial Cuda images and download images to device
    std::cout << "Initializing data..." << std::endl;
    InitCuda(devNum);
    CudaImage img1;
    img1.AllocateUnified(w, h, w, false, NULL, (float *)limg.data);
    img1.Download();

    // Extract Sift features from images
    SiftData siftData1;
    float initBlur = 1.0f;
    float thresh = (imgSet ? 4.5f : 3.0f);
    InitSiftData(siftData1, 32768, true, true);

    float *memoryTmp = AllocSiftTempMemory(w, h, 5, false);
    double average_time = 0;
#define N 100
#define WARMUP 5
    for (int i = 0; i < N; i++) {
        double temp_time = ExtractSift(siftData1, img1, 5, initBlur, thresh,
                                       0.0f, false, memoryTmp);
        if (i >= WARMUP) average_time += temp_time;
    }
    FreeSiftTempMemory(memoryTmp);
    printf("Average Time per run: %.4f\n", average_time / (N - WARMUP));
    PlotKeyPoints(siftData1, img1);
    cv::imwrite("data/limg_pts.pgm", limg);

    // Free Sift data from device
    FreeSiftData(siftData1);
}

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

