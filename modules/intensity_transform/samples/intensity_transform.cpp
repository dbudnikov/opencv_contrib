#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/intensity_transform.hpp"

#include <iostream>
#include <opencv2/core/ocl.hpp>
//#define CL_HPP_ENABLE_EXCEPTIONS
//#include <CL/cl2.hpp>
//#include <CL/opencl.hpp>
#include "opencv2/core/opencl/opencl_info.hpp"

using namespace std;
using namespace cv;
//using namespace cv::ocl;
using namespace cv::intensity_transform;

namespace
{
#if 0
void dumpCLinfo()
{
    std::cout << "*** OpenCL info ***" << std::endl;
    try
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        std::cout << "OpenCL info: Found "  << platforms.size() << " OpenCL platforms" << std::endl;

        for (int i = 0; i < platforms.size(); ++i)
        {
            std::string name = platforms[i].getInfo<CL_PLATFORM_NAME>();
            std::string version = platforms[i].getInfo<CL_PLATFORM_VERSION>();
            std::string profile = platforms[i].getInfo<CL_PLATFORM_PROFILE>();
            std::string extensions = platforms[i].getInfo<CL_PLATFORM_EXTENSIONS>();
            //LOGD( "OpenCL info: Platform[%d] = %s, ver = %s, prof = %s, ext = %s",
            //      i, name.c_str(), version.c_str(), profile.c_str(), extensions.c_str() );
            std::cout << "OpenCL info: Platform[" << i << "] = " << name.c_str()
            << ", ver = " << version.c_str() << ", prof = " << profile.c_str()
            << ", ext = " << extensions.c_str() << std::endl;
        }

        std::vector<cl::Device> devices;
        platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);

        for (int i = 0; i < devices.size(); ++i)
        {
            std::string name = devices[i].getInfo<CL_DEVICE_NAME>();
            std::string extensions = devices[i].getInfo<CL_DEVICE_EXTENSIONS>();
            cl_ulong type = devices[i].getInfo<CL_DEVICE_TYPE>();
            //LOGD( "OpenCL info: Device[%d] = %s (%s), ext = %s",
            //      i, name.c_str(), (type==CL_DEVICE_TYPE_GPU ? "GPU" : "CPU"), extensions.c_str() );
            std::string device_type = (type==CL_DEVICE_TYPE_GPU ? "GPU" : "CPU");
            std::cout << "OpenCL info: Device[" << i << "] = " << name.c_str() << " (" << device_type <<  "), ext = "
            << extensions.c_str() << std::endl;
        }
    }
    catch(const cl::Error& e)
    {
        //LOGE( "OpenCL info: error while gathering OpenCL info: %s (%d)", e.what(), e.err() );
        std::cout << "OpenCL info: error while gathering OpenCL info: " << e.what() << " (" << e.err() << ")" << std::endl;
        throw;
    }
    catch(const std::exception& e)
    {
        //LOGE( "OpenCL info: error while gathering OpenCL info: %s", e.what() );
        std::cout << "OpenCL info: error while gathering OpenCL info: " << e.what() << std::endl;
    }
    catch(...)
    {
        //LOGE( "OpenCL info: unknown error while gathering OpenCL info" );
        std::cout << "OpenCL info: error while gathering OpenCL info" << std::endl;
    }
    std::cout << "*******************" << std::endl;
}
#endif

static std::string keys =
    "{ help  h     | | Print help message. }"
    "{ input i     | | Path to the input image. }";

// global variables
//Mat g_image;
//UMat g_image;
UMat g_image(cv::USAGE_ALLOCATE_DEVICE_MEMORY);

int g_gamma = 40;
const int g_gammaMax = 500;
UMat g_imgGamma(cv::USAGE_ALLOCATE_DEVICE_MEMORY);;
const std::string g_gammaWinName = "Gamma Correction";

UMat g_contrastStretch(cv::USAGE_ALLOCATE_DEVICE_MEMORY);
int g_r1 = 70;
int g_s1 = 15;
int g_r2 = 120;
int g_s2 = 240;
const std::string g_contrastWinName = "Contrast Stretching";

UMat g_imgBIMEF;
int g_mu = 50;
const int g_muMax = 100;
const std::string g_BIMEFWinName = "BIMEF";

static void onTrackbarGamma(int, void*)
{
    float gamma = g_gamma / 100.0f;
    gammaCorrection(g_image, g_imgGamma, gamma);
    imshow(g_gammaWinName, g_imgGamma);
}

static void onTrackbarContrastR1(int, void*)
{
    contrastStretching(g_image, g_contrastStretch, g_r1, g_s1, g_r2, g_s2);
    imshow("Contrast Stretching", g_contrastStretch);
}

static void onTrackbarContrastS1(int, void*)
{
    contrastStretching(g_image, g_contrastStretch, g_r1, g_s1, g_r2, g_s2);
    imshow("Contrast Stretching", g_contrastStretch);
}

static void onTrackbarContrastR2(int, void*)
{
    contrastStretching(g_image, g_contrastStretch, g_r1, g_s1, g_r2, g_s2);
    imshow("Contrast Stretching", g_contrastStretch);
}

static void onTrackbarContrastS2(int, void*)
{
    contrastStretching(g_image, g_contrastStretch, g_r1, g_s1, g_r2, g_s2);
    imshow("Contrast Stretching", g_contrastStretch);
}

static void onTrackbarBIMEF(int, void*)
{
    float mu = g_mu / 100.0f;
    BIMEF(g_image, g_imgBIMEF, mu);
    imshow(g_BIMEFWinName, g_imgBIMEF);
}
}

int main(int argc, char **argv)
{
    CommandLineParser parser(argc, argv, keys);

    const std::string inputFilename = parser.get<String>("input");
    parser.about("Use this script to apply intensity transformation on an input image.");
    if (parser.has("help") || inputFilename.empty())
    {
        parser.printMessage();
        return 0;
    }

    // Read input image
    //g_image = imread(inputFilename);
    Mat tmp_image = imread(inputFilename);
    tmp_image.copyTo(g_image);

    // Create trackbars
    namedWindow(g_gammaWinName);
    createTrackbar("Gamma value", g_gammaWinName, &g_gamma, g_gammaMax, onTrackbarGamma);

    namedWindow(g_contrastWinName);
    createTrackbar("Contrast R1", g_contrastWinName, &g_r1, 256, onTrackbarContrastR1);
    createTrackbar("Contrast S1", g_contrastWinName, &g_s1, 256, onTrackbarContrastS1);
    createTrackbar("Contrast R2", g_contrastWinName, &g_r2, 256, onTrackbarContrastR2);
    createTrackbar("Contrast S2", g_contrastWinName, &g_s2, 256, onTrackbarContrastS2);

    namedWindow(g_BIMEFWinName);
    createTrackbar("Enhancement ratio mu", g_BIMEFWinName, &g_mu, g_muMax, onTrackbarBIMEF);

    // Apply intensity transformations
    UMat imgAutoscaled(cv::USAGE_ALLOCATE_DEVICE_MEMORY), imgLog(cv::USAGE_ALLOCATE_DEVICE_MEMORY);;
    // PERF START
#if 1
    //dumpCLinfo();
    cv::dumpOpenCLInformation();
    if( cv::ocl::useOpenCL() )
    {
        std::cout << "OpenCV+OpenCL works OK!" << std::endl;
    } 
    else
    {
        std::cout << "OpenCV+OpenCL DOESN'T works!" << std::endl;
    }

    size_t loop_length = 100;
    {
        cv::TickMeter tm;
        tm.start();
        for (size_t i = 0; i < loop_length; i++)
        {
            autoscaling(g_image, imgAutoscaled);
        }
        tm.stop();
        std::cout << "autoscaling processed " << loop_length << " frames" << " (" << loop_length / tm.getTimeSec() << " FPS)" << std::endl;
    }
    {
        cv::TickMeter tm;
        tm.start();
        for (size_t i = 0; i < loop_length; i++)
        {
            gammaCorrection(g_image, g_imgGamma, g_gamma/100.0f);
        }
        tm.stop();
        std::cout << "gammaCorrection processed " << loop_length << " frames" << " (" << loop_length / tm.getTimeSec() << " FPS)" << std::endl;
    }
    {
        cv::TickMeter tm;
        tm.start();
        for (size_t i = 0; i < loop_length; i++)
        {
            logTransform(g_image, imgLog);
        }
        tm.stop();
        std::cout << "logTransform processed " << loop_length << " frames" << " (" << loop_length / tm.getTimeSec() << " FPS)" << std::endl;
    }
    {
        cv::TickMeter tm;
        tm.start();
        for (size_t i = 0; i < loop_length; i++)
        {
            contrastStretching(g_image, g_contrastStretch, g_r1, g_s1, g_r2, g_s2);
        }
        tm.stop();
        std::cout << "contrastStretching processed " << loop_length << " frames" << " (" << loop_length / tm.getTimeSec() << " FPS)" << std::endl;
    }
#if 1
    {
        cv::TickMeter tm;
        tm.start();
        for (size_t i = 0; i < loop_length; i++)
        {
            BIMEF(g_image, g_imgBIMEF, g_mu / 100.0f);
        }
        tm.stop();
        std::cout << "BIMEF processed " << loop_length << " frames" << " (" << loop_length / tm.getTimeSec() << " FPS)" << std::endl;
    }
#endif
#endif
//PERF_STOP

    autoscaling(g_image, imgAutoscaled);
    gammaCorrection(g_image, g_imgGamma, g_gamma/100.0f);
    logTransform(g_image, imgLog);
    contrastStretching(g_image, g_contrastStretch, g_r1, g_s1, g_r2, g_s2);
    BIMEF(g_image, g_imgBIMEF, g_mu / 100.0f);

    // Display intensity transformation results
    imshow("Original Image", g_image);
    imshow("Autoscale", imgAutoscaled);
    imshow(g_gammaWinName, g_imgGamma);
    imshow("Log Transformation", imgLog);
    imshow(g_contrastWinName, g_contrastStretch);
    imshow(g_BIMEFWinName, g_imgBIMEF);

    waitKey(0);
    return 0;
}
