//
// Created by wei on 5/22/18.
//

#include "common/common_utils.h"
#include "common/ConfigParser.h"
#include "core/SurfelWarpSerial.h"
#include <boost/filesystem.hpp>

std::vector<surfelwarp::Matrix4f> readMatrix(const std::string& filename, int num_frames)
{
    std::vector<Eigen::Matrix4f> poses;
//    poses.reserve(num_frames);
    std::ifstream file(filename.c_str());
    float number;
    if (file.is_open()) {
        std::string line;
        for (int i = 0; i < num_frames && std::getline(file, line); i++) {
            std::vector<float> myNumbers;
            std::stringstream iss( line );
            while ( iss >> number )
                myNumbers.push_back( number );
            poses.push_back(Eigen::Map<Eigen::Matrix<float,4,4,Eigen::RowMajor>>(myNumbers.data()));
        }
        file.close();
    }

    return poses;
}

int main(int argc, char** argv) {
    using namespace surfelwarp;

    //Get the config path
    std::string config_path, poses_path;
    if (argc <= 1) {
#if defined(WIN32)
        config_path = "C:/Users/wei/Documents/Visual Studio 2015/Projects/surfelwarp/test_data/boxing_config.json";
#else
        config_path = "/home/wei/Documents/programs/surfelwarp/test_data/boxing_config.json";
#endif
    } else {
        config_path = std::string(argv[1]);
        if (argc == 3)
            poses_path = std::string(argv[2]);
    }

    //Parse it
    auto& config = ConfigParser::Instance();
    config.ParseConfig(config_path);

    //The context
    //auto context = initCudaContext();

    std::vector<surfelwarp::Matrix4f> poses;
    if(!poses_path.empty()){
        poses = readMatrix(poses_path, config.num_frames());
        LOG(INFO) << "Read " << poses.size() << " poses";
    }

    auto fetcher = std::make_shared<surfelwarp::VolumeDeformFileFetch>(config.data_path());
    std::cout<<config.data_path()<<std::endl;

    cv::Mat m_depth_img, m_rgb_img;
//    m_depth_img = cv::Mat(cv::Size(config.raw_image_cols(), config.raw_image_rows()), CV_16UC1);
    m_depth_img = cv::Mat(cv::Size(config.raw_image_cols(), config.raw_image_rows()), CV_16FC1);
    m_rgb_img = cv::Mat(cv::Size(config.raw_image_cols(), config.raw_image_rows()), CV_8UC4);


    //The processing loop
    SurfelWarpSerial fusion;

    for(auto i = config.start_frame_idx(); i < config.num_frames(); i++){
        if(config.frame_skip() > 1 && i % config.frame_skip())
            continue;

        if(poses_path.empty())
        {
            fusion.Process(nullptr, nullptr, true);
        }
        else
        {
            fetcher->FetchDepthImage(i, m_depth_img);
            fetcher->FetchRGBImage(i, m_rgb_img);
            fusion.SetPose(poses[i]);
            fusion.Process(&m_rgb_img, &m_depth_img, false);
        }
    }

    //destroyCudaContext(context);
}
