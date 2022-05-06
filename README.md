# TITLE: Optimizing SIFT for an Embedded GPU

##### Members: David Wang (dwang3) Prithvi Suresh (psuresh2)

# [Presentation Link](https://youtu.be/uRTEdcjqfKg)


# SUMMARY: 
We plan to parallelize SIFT using cuda to run on an nvidia-gpu. Although we don’t necessarily know the target speed-up, we are interested in understanding how our approaches to parallelizing would differ based on different target devices. We selected two devices, NVIDIA GeForce RTX 2080 from the Gates Cluster and an NVIDIA Jetson Nano, and will measure the differences in the speedup between the two devices.

# Background: 
With the growing affinity for AI/Robotics in almost all applications, there is an increasing need for faster, smaller, and more power efficient edge devices. The capabilities of these edge devices pale in comparison to large gaming GPUs located in server farms. Nvidia has already tried to bridge this gap by releasing a suite of embedded GPUs. These embedded devices are optimized to run machine learning inference at low power. Developing on these devices is virtually similar to developing on a traditional large Nvidia GPU. Through this project, we aim to explore the effects a resource constrained device can have on parallelizing approaches. To this end we select two devices and an image processing algorithm to parallelize.

Image Processing on edge devices proves beneficial for a wide range of applications such as robotics and surveillance. We choose the Scale Invariant Feature Transform (SIFT) algorithm, a widely known keypoint detection algorithm to parallelize. SIFT is widely used in image processing to extract key points for object detection, localization, and tracking. SIFT also has huge scope for parallelism and can take advantage of the multiple cores of a GPU. The summary of steps involved in SIFT is as follows: 
1. Scale Space Construction: Construct different scales and different blur levels for a single image 
2. LoG: Obtain the Laplacian of the Gaussian of each of these images and extract local minima and maxima. 
3. Threshold extrema: Remove noisy extrema such as edges 
4. Assign Rotation Invariance: Make extrema invariant to rotation 
5. Assign feature: Assign a feature vector to the point

The devices we selected are NVIDIA GeForce RTX 2080 and NVIDIA Jetson Nano. These devices both support cuda and theoretically can run the same piece of code. However, we hope to see large differences in performance when running the same algorithm on them. Essentially, we hypothesize a performance drop when the generally parallelized algorithm is run on the embedded device and we hope to deduce optimization strategies by analyzing the hardware constraints.

# Challenge
Application Level Challenges: We anticipate challenges in effectively parallelizing SIFT. SIFT has multiple steps and we might find it a challenge to reduce the number of kernel launches. Establishing a good baseline on both devices. Hardware Challenges: Inability to achieve satisfactory speed-up on either device.

SIFT has a series of operations such as blurring, extrema detection all which benefit from pixel level parallelism. These can run on a powerful GPU so we expect a significant speed-up in the larger GPU. However, in the smaller GPU, we are constrained by memory and bandwidth.

# RESOURCES
We plan to use the original sequential implementation that openCV provides. This proves to be a decent baseline and will develop our cudaSIFT based on this. We will be using the gates cluster’s gpu to get our first parallized version of SIFT up and running. We will then proceed to deploy it on an embedded GPU and further study and optimize it. The technical specifications for these GPUs can be found here and here 

# GOALS
- 25% - Parallelize SIFT for any Nvidia GPU 
- 50% - Run parallelized SIFT on Jetson Nano 
- 75% - optimize parallelized SIFT on Jetson Nano and compare with original SIFT 
- 100% - Thoroughly analyze optimization and report findings between approaches taken 
- 125% - Run SIFT real-time on an nvidia jetson nano via a webcam

We hope to show a live demonstration of parallelized SIFT by running keypoint detection on the embedded GPU in real-time. If not, we will display our findings for strategies and the difference in optimizing for an embedded gpu and a really large one.

# Getting Started 
### PSC set-up instructions
We need to build opencv from source since the existing installation on bridges was compiled with an older compiler. This causes linking errors since gcc < 5 had a different ABI.

1. Get the newest version of opencv
`wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip` 
`unzip opencv.zip`
2. Create build directory
`mkdir -p build && cd build`
3. Specify target directory for opencv
`cmake -D CMAKE_INSTALL_PREFIX='path/to/target/directory' ../opencv-4.x/`
4. Build and install at target directory
`cmake --build . --target install`
5. Append existing path
`export PATH=/path/to/target/directory:PATH`

### Hello world

1. clone the repo and checkout to the psc branch
`git clone https://github.com/prithusuresh/cudaSIFT.git`
`git checkout psc`

##### Note: for psc machines
Run the rest of the commands in interactive mode or in batch mode. 
1. Load cuda
`module load cuda`
2. `cmake .`
3.  `make`
4.  `./cudaSift jetson.jpg`

This should run the opencv's implementation of SIFT and a series of prints from the GPU.

# MILESTONE

We had initially planned on implementing SIFT using cuda from scratch but later discovered that the latest version of opencv provides implicit support of cuda. To enable this support, we had to rebuild some libraries on the bridges machine. We had initally planned on using the gates cluster but discovered that updating some of the libraries was proving hard. Hence we are shifiting to use bridges2 as part of psc. 

We also discovered a [repo](https://github.com/Celebrandil/CudaSift) with optimized cuda for various gpu architectures. Given that the goal of the assignment was to understand and document the difference in optimizing code for an embedded gpu, we decided to shift our focus to optimizing this existing algorithm for the jetson. After working on running this on the bridges and the jetson nano, we have an implemented baseline. 

Currently, the sequential version of SIFT as per the openCV implementation on an image of size (1920, 1080) takes 272 ms. The cuda version of SIFT on an image of size (1920, 1080) takes 0.47 ms for SIFT on the gpu alone and 0.81 ms along with cudaMemcpy. Times were estimated by taking the average of 200 runs. 

We pretty much seem ontrack for the rest of our goals, but it would be interesting to see if we are able to optimize cudaSIFT for a webcam given that half the time is bottlenecked by memory movement. We are aiming to document a framework for optimizing for embedded devices. For the demo, we'd like to show our findings and optimization strategies along with a small demo. 

Sample keypoint detector on cudaSIFT:
![](/imgs/image.png)

