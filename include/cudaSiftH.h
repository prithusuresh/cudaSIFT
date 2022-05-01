#ifndef CUDASIFTH_H
#define CUDASIFTH_H

#include "cudaImage.h"
#include "cuda_runtime.h"
#include "cudautils.h"
//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//

int ExtractSiftLoop(SiftData &siftData, CudaImage &img, int numOctaves,
                    double initBlur, float thresh, float lowestScale,
                    float subsampling, float *memoryTmp, float *memorySub);
void ExtractSiftOctave(SiftData &siftData, CudaImage &img, int octave,
                       float thresh, float lowestScale, float subsampling,
                       float *memoryTmp);
double ScaleDown(CudaImage &res, CudaImage &src, float variance);
double ScaleUp(CudaImage &res, CudaImage &src);
double ComputeOrientations(cudaTextureObject_t texObj, CudaImage &src,
                           SiftData &siftData, int octave);
double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData &siftData,
                              float subsampling, int octave);
double OrientAndExtract(cudaTextureObject_t texObj, SiftData &siftData,
                        float subsampling, int octave);
double RescalePositions(SiftData &siftData, float scale);
double LowPass(CudaImage &res, CudaImage &src, float scale);
void PrepareLaplaceKernels(int numOctaves, float initBlur, float *kernel);
double LaplaceMulti(cudaTextureObject_t texObj, CudaImage &baseImage,
                    CudaImage *results, int octave);
double FindPointsMulti(CudaImage *sources, SiftData &siftData, float thresh,
                       float edgeLimit, float factor, float lowestScale,
                       float subsampling, int octave);

int ExtractSiftLoopAsync(SiftData &siftData, CudaImage &img, int numOctaves,
                         double initBlur, float thresh, float lowestScale,
                         float subsampling, float *memoryTmp, float *memorySub,
                         cudaStream_t cuda_stream);
void ExtractSiftOctaveAsync(SiftData &siftData, CudaImage &img, int octave,
                            float thresh, float lowestScale, float subsampling,
                            float *memoryTmp, cudaStream_t cuda_stream);
double ScaleDownAsync(CudaImage &res, CudaImage &src, float variance,
                      cudaStream_t cuda_stream);
double ScaleUpAsync(CudaImage &res, CudaImage &src, cudaStream_t cuda_stream);
double ComputeOrientationsAsync(cudaTextureObject_t texObj, CudaImage &src,
                                SiftData &siftData, int octave,
                                cudaStream_t cuda_stream);
double ExtractSiftDescriptorsAsync(cudaTextureObject_t texObj,
                                   SiftData &siftData, float subsampling,
                                   int octave, cudaStream_t cuda_stream);
double OrientAndExtractAsync(cudaTextureObject_t texObj, SiftData &siftData,
                             float subsampling, int octave,
                             cudaStream_t cuda_stream);
double RescalePositionsAsync(SiftData &siftData, float scale,
                             cudaStream_t cuda_stream);
double LowPassAsync(CudaImage &res, CudaImage &src, float scale,
                    cudaStream_t cuda_stream);
void PrepareLaplaceKernelsAsync(int numOctaves, float initBlur, float *kernel,
                                cudaStream_t cuda_stream);
double LaplaceMultiAsync(cudaTextureObject_t texObj, CudaImage &baseImage,
                         CudaImage *results, int octave,
                         cudaStream_t cuda_stream);
double FindPointsMultiAsync(CudaImage *sources, SiftData &siftData,
                            float thresh, float edgeLimit, float factor,
                            float lowestScale, float subsampling, int octave,
                            cudaStream_t cuda_stream);
#endif
