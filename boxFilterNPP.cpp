/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>

#include <opencv2/opencv.hpp>
#include <cufft.h>

#include "device_launch_parameters.h"
#include <helper_functions.h>
#include <cmath>

using namespace std;

// Global boolean variable to control verbosity
bool verbose = true;

// Kernel to block frequencies on the host
void blockFrequenciesHost(cufftComplex *data, int width, int height, int blockWidth, int blockHeight)
{
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      if (x >= width - blockWidth || y >= height - blockHeight)
      {
        int index = y * width + x;
        data[index].x = 0;
        data[index].y = 0;
      }
    }
  }
}

void saveComplexSquares(const cufftComplex *h_outputData, int width, int height, const std::string &name)
{
  // Allocate memory for squared real and imaginary parts
  float *realSquares = new float[width * height];
  float *imagSquares = new float[width * height];

  // Compute squares of real and imaginary parts
  for (int i = 0; i < height; ++i)
  {
    for (int j = 0; j < width; ++j)
    {
      int index = i * width + j;
      realSquares[index] = h_outputData[index].x * h_outputData[index].x;
      imagSquares[index] = h_outputData[index].y * h_outputData[index].y;
    }
  }

  // Create OpenCV matrices for real and imaginary squares
  cv::Mat realSquareMat(height, width, CV_32FC1, realSquares);
  cv::Mat imagSquareMat(height, width, CV_32FC1, imagSquares);

  // Normalize the matrices for visualization
  cv::normalize(realSquareMat, realSquareMat, 0, 255, cv::NORM_MINMAX);
  cv::normalize(imagSquareMat, imagSquareMat, 0, 255, cv::NORM_MINMAX);

  // Convert to 8-bit unsigned integer
  realSquareMat.convertTo(realSquareMat, CV_8U);
  imagSquareMat.convertTo(imagSquareMat, CV_8U);

  // Save real and imaginary squares as PNG files
  cv::imwrite(name + "_realSQR.png", realSquareMat);
  cv::imwrite(name + "_imagSQR.png", imagSquareMat);

  // Clean up
  delete[] realSquares;
  delete[] imagSquares;
}

void createVerticalSineSquared(const std::string &name, int period)
{
  // Create a 512x512 matrix
  cv::Mat image(512, 512, CV_8UC1);

  // Fill the image with sine squared values in the x-direction
  for (int x = 0; x < image.cols; ++x)
  {
    double sineSquared = std::sin(2 * M_PI * x / period);
    sineSquared *= sineSquared; // Square the sine value
    for (int y = 0; y < image.rows; ++y)
    {
      image.at<uchar>(y, x) = static_cast<uchar>(sineSquared * 255); // Convert to uchar
    }
  }

  // Save the image as PNG
  cv::imwrite(name + ".png", image);
}

// Function to print text if verbose is true
void Text(const std::string &text)
{
  if (verbose)
  {
    cout << text << endl;
  }
}

bool printfNPPinfo(int argc, char *argv[])
{
  const NppLibraryVersion *libVer = nppGetLibVersion();

  printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
         libVer->build);

  int driverVersion, runtimeVersion;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
         (driverVersion % 100) / 10);
  printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
         (runtimeVersion % 100) / 10);

  // Min spec is SM 1.0 devices
  bool bVal = checkCudaCapabilities(1, 0);
  return bVal;
}

int main(int argc, char *argv[])
{
  printf("%s Starting...\n\n", argv[0]);

  try
  {
    std::string sFilename;
    char *filePath;

    findCudaDevice(argc, (const char **)argv);

    if (printfNPPinfo(argc, argv) == false)
    {
      exit(EXIT_SUCCESS);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "input"))
    {
      getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
    }
    else
    {
      filePath = sdkFindFilePath("Lena.pgm", argv[0]);
    }

    if (filePath)
    {
      sFilename = filePath;
    }
    else
    {
      sFilename = "Lena.pgm";
    }

    // if we specify the filename at the command line, then we only test
    // sFilename[0].
    int file_errors = 0;
    std::ifstream infile(sFilename.data(), std::ifstream::in);

    if (infile.good())
    {
      std::cout << "boxFilterNPP opened: <" << sFilename.data()
                << "> successfully!" << std::endl;
      file_errors = 0;
      infile.close();
    }
    else
    {
      std::cout << "boxFilterNPP unable to open: <" << sFilename.data() << ">"
                << std::endl;
      file_errors++;
      infile.close();
    }

    if (file_errors > 0)
    {
      exit(EXIT_FAILURE);
    }

    std::string sResultFilename = sFilename;

    std::string::size_type dot = sResultFilename.rfind('.');

    if (dot != std::string::npos)
    {
      sResultFilename = sResultFilename.substr(0, dot);
    }

    sResultFilename += "_boxFilter.pgm";

    if (checkCmdLineFlag(argc, (const char **)argv, "output"))
    {
      char *outputFilePath;
      getCmdLineArgumentString(argc, (const char **)argv, "output",
                               &outputFilePath);
      sResultFilename = outputFilePath;
    }

    // declare a host image object for an 8-bit grayscale image
    npp::ImageCPU_8u_C1 oHostSrc;
    // load gray-scale image from disk
    npp::loadImage(sFilename, oHostSrc);
    // declare a device image and copy construct from the host image,
    // i.e. upload host to device
    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

    // create struct with box-filter mask size
    NppiSize oMaskSize = {5, 5};

    NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    NppiPoint oSrcOffset = {0, 0};

    // create struct with ROI size
    NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    // allocate device image of appropriately reduced size
    npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);
    // set anchor point inside the mask to (oMaskSize.width / 2,
    // oMaskSize.height / 2) It should round down when odd
    NppiPoint oAnchor = {oMaskSize.width / 2, oMaskSize.height / 2};

    // run box filter
    NPP_CHECK_NPP(nppiFilterBoxBorder_8u_C1R(
        oDeviceSrc.data(), oDeviceSrc.pitch(), oSrcSize, oSrcOffset,
        oDeviceDst.data(), oDeviceDst.pitch(), oSizeROI, oMaskSize, oAnchor,
        NPP_BORDER_REPLICATE));

    // declare a host image for the result
    npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    // and copy the device result data into it
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

    saveImage(sResultFilename, oHostDst);
    std::cout << "Saved image: " << sResultFilename << std::endl;

    // verified
    // Redo Lena *.png from PGMs
    cv::Mat img_Lena_orig = imread("Lena.pgm", cv::IMREAD_UNCHANGED);
    cv::imwrite("./Lena_orig.png", img_Lena_orig);
    cv::Mat img_Lena_boxFilter = imread("Lena_boxFilter.pgm", cv::IMREAD_UNCHANGED);
    cv::imwrite("./Lena_postBox.png", img_Lena_boxFilter);

    nppiFree(oDeviceSrc.data());
    nppiFree(oDeviceDst.data());

    // ************ FFT processing *************************

    // Load the original image using OpenCV
    cv::Mat originalImage = cv::imread("Lena_orig.png", cv::IMREAD_GRAYSCALE);
    if (originalImage.empty())
    {
      std::cerr << "Error: Failed to load original image." << std::endl;
      return 1;
    }

    // Get image dimensions
    int width = originalImage.cols;
    int height = originalImage.rows;

    // Allocate memory for input and output data on the GPU
    cufftReal *d_inputData;
    cufftComplex *d_outputData;
    cudaMalloc(&d_inputData, width * height * sizeof(cufftReal));
    cudaMalloc(&d_outputData, (width / 2 + 1) * height * sizeof(cufftComplex)); // Only store half of the complex numbers for real input

    // Convert image data to cufftReal format and copy to GPU memory
    cufftReal *h_imageData = new cufftReal[width * height];
    for (int i = 0; i < height; ++i)
    {
      for (int j = 0; j < width; ++j)
      {
        h_imageData[i * width + j] = static_cast<cufftReal>(originalImage.at<unsigned char>(i, j));
      }
    }
    cudaMemcpy(d_inputData, h_imageData, width * height * sizeof(cufftReal), cudaMemcpyHostToDevice);
    delete[] h_imageData;

    // Create a forward FFT plan
    cufftHandle forwardPlan;
    cufftPlan2d(&forwardPlan, height, width, CUFFT_R2C);

    // Execute forward FFT
    cufftExecR2C(forwardPlan, d_inputData, d_outputData);

    // Copy the FFT data back to host memory
    cufftComplex *h_outputDataUnblocked = new cufftComplex[(width / 2 + 1) * height];
    cudaMemcpy(h_outputDataUnblocked, d_outputData, (width / 2 + 1) * height * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    // void saveComplexSquares(const cufftComplex *h_outputData, int width, int height, const std::string &name)
    saveComplexSquares(h_outputDataUnblocked, (width / 2 + 1), height, "Freq_Lena_Unblocked");

    // Define the bands to block (right and bottom)
    int blockWidth = static_cast<int>(0.1 * width);     // 10% of width
    int blockHeight = static_cast<int>(0.1 * height);   // 10% of height

    // Allocate memory for blocked output data on the host
    cufftComplex *h_outputDataBlocked = new cufftComplex[(width / 2 + 1) * height];
    cudaMemcpy(h_outputDataBlocked, h_outputDataUnblocked, (width / 2 + 1) * height * sizeof(cufftComplex), cudaMemcpyHostToHost);

    // Apply block filter on the host
    blockFrequenciesHost(h_outputDataBlocked, width / 2 + 1, height, blockWidth, blockHeight);

    // Copy the filtered FFT data back to device memory
    cudaMemcpy(d_outputData, h_outputDataBlocked, (width / 2 + 1) * height * sizeof(cufftComplex), cudaMemcpyHostToDevice);

    // Perform inverse FFT on the blocked data
    cufftHandle inversePlan;
    cufftPlan2d(&inversePlan, height, width, CUFFT_C2R);
    cufftExecC2R(inversePlan, d_outputData, d_inputData);

    // Copy reconstructed image data back to host memory
    cufftReal *h_reconstructedDataBlocked = new cufftReal[width * height];
    cudaMemcpy(h_reconstructedDataBlocked, d_inputData, width * height * sizeof(cufftReal), cudaMemcpyDeviceToHost);

    // Normalize the reconstructed image to the range [0, 255]
    cv::Mat reconstructedImageBlocked(height, width, CV_32FC1, h_reconstructedDataBlocked);
    cv::normalize(reconstructedImageBlocked, reconstructedImageBlocked, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // Save the blocked reconstructed image as a PNG file
    cv::imwrite("reconstructed_image_blocked.png", reconstructedImageBlocked);

    // void createVerticalSineSquared(const std::string &name, int period)
    //createVerticalSineSquared("sin_Per_20", 20);



    // Clean up
    delete[] h_outputDataUnblocked;
    delete[] h_outputDataBlocked;
    delete[] h_reconstructedDataBlocked;
    cufftDestroy(forwardPlan);
    cufftDestroy(inversePlan);
    cudaFree(d_inputData);
    cudaFree(d_outputData);

    // repeat above analysis for periodic vertical sin bars

    // Load the original image using OpenCV
    cv::Mat originalImage2 = cv::imread("sin_Per_20.png", cv::IMREAD_GRAYSCALE);
    if (originalImage2.empty())
    {
      std::cerr << "Error: Failed to load original image." << std::endl;
      return 1;
    }

    // Get image dimensions
    int width2 = originalImage.cols;
    int height2 = originalImage.rows;

    // Allocate memory for input and output data on the GPU
    cufftReal *d_inputData2;
    cufftComplex *d_outputData2;
    cudaMalloc(&d_inputData2, width2 * height2 * sizeof(cufftReal));
    cudaMalloc(&d_outputData2, (width2 / 2 + 1) * height2 * sizeof(cufftComplex)); // Only store half of the complex numbers for real input

    // Convert image data to cufftReal format and copy to GPU memory
    cufftReal *h_imageData2 = new cufftReal[width2 * height2];
    for (int i = 0; i < height2; ++i)
    {
      for (int j = 0; j < width2; ++j)
      {
        h_imageData2[i * width2 + j] = static_cast<cufftReal>(originalImage2.at<unsigned char>(i, j));
      }
    }
    cudaMemcpy(d_inputData2, h_imageData2, width2 * height2 * sizeof(cufftReal), cudaMemcpyHostToDevice);
    delete[] h_imageData2;

    // Create a forward FFT plan
    cufftHandle forwardPlan2;
    cufftPlan2d(&forwardPlan2, height2, width2, CUFFT_R2C);

    // Execute forward FFT
    cufftExecR2C(forwardPlan2, d_inputData2, d_outputData2);

    // Copy the FFT data back to host memory
    cufftComplex *h_outputDataUnblocked2 = new cufftComplex[(width / 2 + 1) * height];
    cudaMemcpy(h_outputDataUnblocked2, d_outputData2, (width2 / 2 + 1) * height2 * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    // void saveComplexSquares(const cufftComplex *h_outputData, int width, int height, const std::string &name)
    saveComplexSquares(h_outputDataUnblocked2, (width2 / 2 + 1), height2, "FreqSpect_Sin2_P40");

    // ************end FFT processing ******************

    exit(EXIT_SUCCESS);
    }
  catch (npp::Exception &rException)
  {
    std::cerr << "Program error! The following exception occurred: \n";
    std::cerr << rException << std::endl;
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
  }
  catch (...)
  {
    std::cerr << "Program error! An unknow type of exception occurred. \n";
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
    return -1;
  }

  return 0;
}
