// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

using namespace cv;
using namespace std;

#include <iostream>

#define LOG_TRANSFORM_PERF_PROFILE

namespace cv {
namespace intensity_transform {

void logTransform(const UMat input, UMat& output)
{
#ifdef LOG_TRANSFORM_PERF_PROFILE
    cv::TickMeter tm_all, tm_minmax, tm_log, tm_cvt1, tm_cvt2;
    tm_all.start();
#endif

    double maxVal;
#ifdef LOG_TRANSFORM_PERF_PROFILE
    tm_minmax.start();
#endif
    minMaxLoc(input, NULL, &maxVal, NULL, NULL);
#ifdef LOG_TRANSFORM_PERF_PROFILE   
    tm_minmax.stop();
#endif

    const double c = 255 / log(1 + maxVal);
    UMat add_one_64f(cv::USAGE_ALLOCATE_DEVICE_MEMORY);
#ifdef LOG_TRANSFORM_PERF_PROFILE
    tm_cvt1.start();
#endif
    input.convertTo(add_one_64f, CV_64F, 1, 1.0f);
#ifdef LOG_TRANSFORM_PERF_PROFILE
    tm_cvt1.stop();
#endif
    UMat log_64f(cv::USAGE_ALLOCATE_DEVICE_MEMORY);
#ifdef LOG_TRANSFORM_PERF_PROFILE
    tm_log.start();
#endif
    cv::log(add_one_64f, log_64f);
#ifdef LOG_TRANSFORM_PERF_PROFILE
    tm_log.stop();
#endif
#ifdef LOG_TRANSFORM_PERF_PROFILE
    tm_cvt2.start();
#endif
    log_64f.convertTo(output, CV_8UC3, c, 0.0f);
#ifdef LOG_TRANSFORM_PERF_PROFILE
    tm_cvt2.stop();
#endif
#ifdef LOG_TRANSFORM_PERF_PROFILE   
    tm_all.stop();
    std::cout << "logTransform ALL " << tm_all.getTimeSec() << " sec)" << std::endl;
    std::cout << "logTransform minMaxLoc " << tm_minmax.getTimeSec() << " sec)" << std::endl;
    std::cout << "logTransform input.convertTo add_one_64f " << tm_cvt1.getTimeSec() << " sec)" << std::endl;
    std::cout << "logTransform cv::log " << tm_log.getTimeSec() << " sec)" << std::endl;
    std::cout << "logTransform log_64f.convertTo output " << tm_cvt2.getTimeSec() << " sec)" << std::endl;
#endif
}

void gammaCorrection(const UMat input, UMat& output, const float gamma)
{
    std::array<uchar, 256> table;
    for (int i = 0; i < 256; i++)
    {
        table[i] = saturate_cast<uchar>(pow((i / 255.0), gamma) * 255.0);
    }

    LUT(input, table, output);
}

void autoscaling(const UMat input, UMat& output)
{
    double minVal, maxVal;
    minMaxLoc(input, &minVal, &maxVal, NULL, NULL);
    //output = 255 * (input - minVal) / (maxVal - minVal);
    double multiplier = 255.0 / (maxVal - minVal);
    UMat sub_img(cv::USAGE_ALLOCATE_DEVICE_MEMORY); 
    cv::subtract(input, minVal, sub_img);
    cv::multiply(sub_img, multiplier, output);
}

void contrastStretching(const UMat input, UMat& output, const int r1, const int s1, const int r2, const int s2)
{
    std::array<uchar, 256> table;
    for (int i = 0; i < 256; i++)
    {
        if (i <= r1)
        {
            table[i] = saturate_cast<uchar>(((float)s1 / (float)r1) * i);
        }
        else if (r1 < i && i <= r2)
        {
            table[i] = saturate_cast<uchar>(((float)(s2 - s1)/(float)(r2 - r1)) * (i - r1) + s1);
        }
        else // (r2 < i)
        {
            table[i] = saturate_cast<uchar>(((float)(255 - s2)/(float)(255 - r2)) * (i - r2) + s2);
        }
    }

    LUT(input, table, output);
}

}} // cv::intensity_transform::