#include "ros/ros.h"
#include "std_msgs/String.h"



#include <videoSource.h>
#include <videoOutput.h>
#include <commandLine.h>
#include "nvAprilTags.h"
#include <imageFormat.h>

#include <signal.h>
#include <memory>
#include <string>
#include <vector>

#include <iostream>
#include <sstream>
#include <ctime>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include "cuda.h"		  // NOLINT - include .h without directory
#include "cuda_runtime.h" // NOLINT - include .h without directory


bool signal_recieved = false;

void sig_handler(int signo)
{
	if (signo == SIGINT)
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}
struct CailbrateData
{
	cv::Size imageSize;
	cv::Mat cameraMatrix;
	cv::Mat distCoeffs;
	std::vector<std::vector<cv::Point2f>> imagePoints;
	std::string calibration_time;
	float grid_width;
	void read(const cv::FileNode &fs)
	{
		if (fs.empty())
		{
			std::cerr << "FileNode Read Error\n";
			return;
		}
		// printf("%d\n", fs.size());
		fs["calibration_time"] >> calibration_time;
		fs["cameraMatrix"] >> cameraMatrix;
		fs["image_width"] >> imageSize.width;
		fs["image_height"] >> imageSize.height;
		fs["distortion_coefficients"] >> distCoeffs;
	}
	void read(const cv::FileStorage &fs)
	{
		// calibration_time=(std::string)(fs["calibration_time"]);
		fs["calibration_time"] >> calibration_time;
		fs["camera_matrix"] >> cameraMatrix;
		fs["image_width"] >> imageSize.width;
		fs["image_height"] >> imageSize.height;
		fs["distortion_coefficients"] >> distCoeffs;
	}
	void Print()
	{
		std::cout << "calibration_time\t" << calibration_time
				  << "\nimage_width\t\t" << imageSize.width
				  << "\nimage_height\t\t" << imageSize.height << std::endl;
		std::cout << cameraMatrix << std::endl;
		std::cout << distCoeffs << std::endl;
	}
};

struct AprilTagsInfo
{
	float tag_edge_size_;
	size_t max_tags_;
};

struct AprilTagsImpl
{
	// Handle used to interface with the stereo library.
	nvAprilTagsHandle april_tags_handle = nullptr;

	// Camera intrinsics
	nvAprilTagsCameraIntrinsics_t cam_intrinsics;

	// Output vector of detected Tags
	std::vector<nvAprilTagsID_t> tags;

	// CUDA stream
	cudaStream_t main_stream = {};

	// CUDA buffers to store the input image.
	nvAprilTagsImageInput_t input_image;

	// CUDA memory buffer container for RGBA images.
	char *input_image_buffer = nullptr;

	// Size of image buffer
	size_t input_image_buffer_size = 0;

	void initialize(const AprilTagsInfo &node, const CailbrateData &cali, const uint32_t width,
					const uint32_t height, const size_t image_buffer_size,
					const size_t pitch_bytes)
	{
		// std::assert(april_tags_handle == nullptr && "Already initialized.");

		// Get camera intrinsics
		// const double * k = msg_ci->k.data();
		// const float fx = static_cast<float>(k[0]);
		// const float fy = static_cast<float>(k[4]);
		// const float cx = static_cast<float>(k[2]);
		// const float cy = static_cast<float>(k[5]);
		// std::cout<<"A\n";
		const float fx = (float)*(double *)(cali.cameraMatrix.row(0).col(0).data);
		const float fy = (float)*(double *)(cali.cameraMatrix.row(1).col(1).data);
		const float cx = (float)*(double *)(cali.cameraMatrix.row(0).col(2).data);
		const float cy = (float)*(double *)(cali.cameraMatrix.row(1).col(2).data);
		cam_intrinsics = {fx, fy, cx, cy};
		// std::cout<<"B\n";
		// Create AprilTags detector instance and get handle
		const int error = nvCreateAprilTagsDetector(
			&april_tags_handle, width, height, nvAprilTagsFamily::NVAT_TAG36H11,
			&cam_intrinsics, node.tag_edge_size_);
		if (error != 0)
		{
			throw std::runtime_error(
				"Failed to create NV April Tags detector (error code " +
				std::to_string(error) + ")");
		}
		// std::cout<<"C\n";

		// Create stream for detection
		cudaStreamCreate(&main_stream);
		// std::cout<<"D\n";
		// Allocate the output vector to contain detected AprilTags.
		tags.resize(node.max_tags_);
		// std::cout<<"E\n";
		// Setup input image CUDA buffer.
		// const cudaError_t cuda_error =
		// 	cudaMalloc(&input_image_buffer, image_buffer_size);
		// if (cuda_error != cudaSuccess)
		// {
		// 	throw std::runtime_error(
		// 		"Could not allocate CUDA memory (error code " +
		// 		std::to_string(cuda_error) + ")");
		// }
		// std::cout<<"F\n";
		// Setup input image.
		input_image_buffer_size = image_buffer_size;
		input_image.width = width;
		input_image.height = height;
		input_image.dev_ptr = reinterpret_cast<uchar4 *>(input_image_buffer);
		input_image.pitch = pitch_bytes;

		// std::cout<<"G\n";
	}

	~AprilTagsImpl()
	{
		if (april_tags_handle != nullptr)
		{
			cudaStreamDestroy(main_stream);
			nvAprilTagsDestroy(april_tags_handle);
			cudaFree(input_image_buffer);
		}
	}
};


int main(int argc, char **argv)
{
    ros::init(argc, argv, "nvcsiapriltag");
    ros::NodeHandle nh;
    ROS_INFO("Start CSI Camera Apriltag Detected");
    ros::Rate loop_rate(1000);
	ros::Publisher pub=nh.advertise<nvcsiapriltag::AprilTagDectionArray>("apriltag_handle",60);
	nvcsiapriltag::AprilTagDectionArray msg;
    CailbrateData caildata;
	AprilTagsInfo aprilti;
	videoOptions opt;
	std::unique_ptr<AprilTagsImpl> impl_(std::make_unique<AprilTagsImpl>());
    if (signal(SIGINT, sig_handler) == SIG_ERR)
		ROS_ERROR("can't catch SIGINT");
    nh.param<size_t>("max_tag_num", aprilti.max_tags_, (size_t)20);
	nh.param<float>("tag_decimate", aprilti.tag_edge_size_, 1.0f);
	nh.param<uint32_t>("width", opt.width, (uint32_t)1280);
	nh.param<uint32_t>("height", opt.height, (uint32_t)720);
	nh.param<float>("framerate", opt.frameRate, 60.0f);
	std::string flipmethod;
	nh.param<std::string>("filpmethod", flipmethod, 0);
	opt.FlipMethodFromStr(flipmethod.data());
    nh.param<bool>("zerocopy", opt.zeroCopy, false);
	// std::cout << "Camera Info:\nWidth = " << opt.width << "\nHeight = " << opt.height << "\nFrameRate = " << opt.frameRate << "\n"
	// 		  << opt.FlipMethodToStr << std::endl;
	ROS_INFO("Apriltag INFO: max_tag_num = %d     tag_decimate = %d", aprilti.max_tags_, aprilti.tag_edge_size_);
	ROS_INFO("Camera_Info:\nWidth = %d\n Height = %d\nFrameRate = %d\n, FlipMethod = %s\n", opt.width, opt.height, opt.frameRate, opt.FlipMethodToStr);
	videoSource *input = videoSource::Create("csi://0", opt);
    if (!input)
	{
        ROS_ERROR("Error: Failed to create input stream");
		exit(-1);
	}
    videoOutput *output = videoOutput::Create("display://0");
	if (!output)
	{
		ROS_ERROR("Error: Failed to create output stream");
		delete input;
		exit(-2);
	}
    unsigned int framecnt = 0;
    while(ros::ok&&(!signal_recieved))
    {
        uchar4 *imgRGBA = NULL;
        if (!input->Capture(&imgRGBA, 1000))
		{
			if (!input->IsStreaming())
				break;
            ROS_ERROR("failed to capture next frame");
			continue;
		}
        if (impl_->april_tags_handle == nullptr)
		{
			cv::cuda::GpuMat img_rgba8(input->GetHeight(), input->GetWidth(), CV_8UC4, imgRGBA);
			impl_->initialize(aprilti, caildata, input->GetWidth(), input->GetHeight(), img_rgba8.size().width * img_rgba8.size().height * img_rgba8.elemSize(), img_rgba8.step1());
		}
        impl_->input_image_buffer = (char *)imgRGBA;
		impl_->input_image.dev_ptr = reinterpret_cast<uchar4 *>(imgRGBA);
		uint32_t num_detections;
		const int error = nvAprilTagsDetect(
			impl_->april_tags_handle, &(impl_->input_image), impl_->tags.data(),
			&num_detections, aprilti.max_tags_, impl_->main_stream);
		if (error != 0)
		{
			ROS_INFO("Failed to run AprilTags detector (error code %d)", error);
			return -1;
		}
        if (output != NULL)
		{
			static int cnt = 0;
			static float sum = 0;
			output->Render(imgRGBA, input->GetWidth(), input->GetHeight());
			sum += output->GetFrameRate();
			// update status bar
			if (cnt == 9)
			{
				static char str[256];
				sprintf(str, "Camera Viewer (%ux%u) | %.0f FPS %d", input->GetWidth(), input->GetHeight(), sum /= 10, num_detections);
				output->SetStatus(str);
				if (num_detections)
				{
					// std::cout << "Frame : " << framecnt << "\t Apriltag num : " << num_detections << std::endl;
					for (int i = 0; i < num_detections; ++i)
					{
						if(impl_->tags[i].id!=2)
							continue;
						// std::cout << impl_->tags[i].id << "\n";
						for (int j = 0; j < 3; ++j)
							std::cout << impl_->tags[i].translation[j] << "\t\t";
						std::cout << std::endl;
						for(int j=0;j<4;++j)
						{
							std::cout<<impl_->tags[i].corners[j].x<<" \t "<<impl_->tags[i].corners[j].y<<std::endl;
						}
					}
				}
				sum = 0;
				cnt = 0;
			}
			else
				++cnt;

			// check if the user quit
			if (!output->IsStreaming())
				signal_recieved = true;
		}
        ros::spinOnce();
        loop_rate.sleep();
    }
    /*
	 * destroy resources
	 */
	ROS_INFO("\ncamera-viewer:  shutting down...\n");

	SAFE_DELETE(input);
	SAFE_DELETE(output);

	ROS_INFO("camera-viewer:  shutdown complete.\n");
    return 0;
}