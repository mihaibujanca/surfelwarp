
#include "imgproc/frameio/AzureKinectDKFetch.h"
#include "visualization/Visualizer.h"
#include "imgproc/ImageProcessor.h"

using namespace surfelwarp;

int main(int argc, char** argv){

	std::string config_path;
    
    if (argc <= 1) {
        config_path = "/data/surfelwarp/test_data/kinectdk_config.json";
	}else{
        config_path = std::string(argv[1]);
    }
	//Parse it
	auto& config = ConfigParser::Instance();
	config.ParseConfig(config_path);


    Visualizer::Ptr visualizer = std::make_shared<Visualizer>();

    FetchInterface::Ptr image_fetcher = std::make_shared<AzureKinectDKFetch>();
    ImageProcessor::Ptr image_processor = std::make_shared<ImageProcessor>(image_fetcher);


    //cv::Mat depth_img, rgb_img, prev_rgb_img;
    for (int frame_idx=0; frame_idx<10; frame_idx++){

        image_processor->FetchFrame(frame_idx);
        visualizer->DrawDepthImage(image_processor->RawDepthImageCPU());
        cv::imshow("rgb", image_processor->RawRGBImageCPU());
        cv::imshow("rgb_prev", image_processor->RawRGBImagePrevCPU());
        cv::waitKey(0);
    }

    return 0;
}
