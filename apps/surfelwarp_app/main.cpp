//
// Created by wei on 5/22/18.
//

#include "common/common_utils.h"
#include "common/ConfigParser.h"
#include "core/SurfelWarpSerial.h"
#include <boost/filesystem.hpp>

std::vector<surfelwarp::Matrix4f> readMatrix(const char *filename, int num_frames)
{
    std::vector<Eigen::Matrix4f> poses;
    poses.reserve(num_frames);
    std::ifstream file(filename);
    float number;
    if (file.is_open()) {
        std::string line;
        for (int i = 0; i < num_frames && std::getline(file, line); i++) {
            std::vector<float> myNumbers;
            std::stringstream iss( line );
            while ( iss >> number )
                myNumbers.push_back( number );
            poses[i] = Eigen::Map<Eigen::Matrix<float,4,4,Eigen::RowMajor>>(myNumbers.data());
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

	//Save offline
	bool offline_rendering = true;

    std::vector<surfelwarp::Matrix4f> poses;
    if(!poses_path.empty())
        poses = readMatrix(poses_path.c_str(), config.num_frames());

	//The processing loop
	SurfelWarpSerial fusion;

	fusion.ProcessFirstFrame();
	for(auto i = config.start_frame_idx(); i < config.num_frames(); i++){
	    if(config.frame_skip() > 0 && i % config.frame_skip())
	        continue;
//		LOG(INFO) << "The " << i << "th Frame";
                if(poses_path.empty())
                {
                    fusion.ProcessNextFrameWithReinit(offline_rendering);
                }
                else
                {
                    fusion.SetPose(poses[i]);
                    fusion.Process();
                }
	}
	
	//destroyCudaContext(context);
}
