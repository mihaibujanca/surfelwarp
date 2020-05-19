#include "common/common_types.h"
#include "common/common_utils.h"
#include "common/ConfigParser.h"
#include "common/sanity_check.h"
#include "common/CameraObservation.h"
#include "visualization/Visualizer.h"
#include "imgproc/frameio/FetchInterface.h"
#include "imgproc/frameio/GenericFileFetch.h"
#include "imgproc/frameio/AzureKinectDKFetch.h"
#include "imgproc/frameio/VolumeDeformFileFetch.h"
#include "imgproc/ImageProcessor.h"

#include <thread>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>



/*
void testBoundary()
{
	using namespace surfelwarp;
	//Parpare the test data
	auto parser = ConfigParser::Instance();

	//First test fetching
	FileFetch::Ptr fetcher = std::make_shared<FileFetch>(parser.data_path());
	ImageProcessor::Ptr processor = std::make_shared<ImageProcessor>(fetcher);
	
	//CameraObservation observation;
	//processor->ProcessFrameSerial(observation, 130);
	//Visualizer::DrawColoredPointCloud(observation.vertex_confid_map, observation.color_time_map);

	//Process using the observation interface
	for(auto frame_idx = 10; frame_idx < 160; frame_idx++) {
		LOG(INFO) << "Current frame is " << frame_idx;
		CameraObservation observation;
		processor->ProcessFrameSerial(observation, frame_idx);
		if(frame_idx == 130) {
			Visualizer::DrawColoredPointCloud(observation.vertex_confid_map, observation.color_time_map);
		}
	}
}*/

void testFullProcessing() {
	using namespace surfelwarp;

	//First test fetching
	//FileFetch::Ptr fetcher = std::make_shared<FileFetch>(parser.data_path());
	//ImageProcessor::Ptr processor = std::make_shared<ImageProcessor>(fetcher);
	//Get the config path
	std::string config_path;
#if defined(WIN32)
	config_path = "C:/Users/wei/Documents/Visual Studio 2015/Projects/surfelwarp/test_data/boxing_config.json";
#else
	config_path = "/home/xt/Documents/data/surfelwarp/test_data/boxing_config.json";
#endif

	auto& config = ConfigParser::Instance();
	config.ParseConfig(config_path);
	//Construct the image processor
	FetchInterface::Ptr fetcher;
	
	if(config.getIOMode() == "GenericFileFetch") {
		fetcher = std::make_shared<GenericFileFetch>(config.data_path());
	}else if(config.getIOMode() == "VolumeDeformFileFetch"){
		fetcher = std::make_shared<VolumeDeformFileFetch>(config.data_path());
	}else if (config.getIOMode() == "kinect_dk"){
		fetcher = std::make_shared<AzureKinectDKFetch>(config.data_path(), config.isSaveOnlineFrame());
	}else{
		throw(std::runtime_error(config.getIOMode() + " io_mode not supported"));
	}
	ImageProcessor::Ptr processor = std::make_shared<ImageProcessor>(fetcher);
	
	//Do it
	CameraObservation observation;
	for(auto i = config.start_frame_idx(); i < config.num_frames(); i++){
		LOG(INFO) << "The " << i << "th Frame";
		//processor->ProcessFrameSerial(observation, i);
		processor->ProcessFrameStreamed(observation, i);
	}
	//Draw it
	auto draw_func = [&]() {
		//Visualizer::DrawPointCloud(observation.vertex_confid_map);
		Visualizer::DrawSegmentMask(observation.foreground_mask, observation.normalized_rgba_map, 1);
		//Visualizer::DrawGrayScaleImage(observation.filter_foreground_mask);
	};
	
	//Use thread to draw it
	//std::thread draw_thread(draw_func);
	//draw_thread.join();
}

int main() {
	testFullProcessing();
}