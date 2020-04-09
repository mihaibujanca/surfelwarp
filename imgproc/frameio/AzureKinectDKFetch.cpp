#include "AzureKinectDKFetch.h"

surfelwarp::AzureKinectDKFetch::AzureKinectDKFetch(const path& data_path, bool save_online_frame)
{
    const uint32_t deviceCount = k4a::device::get_installed_count();
    if (deviceCount == 0)
    {
        std::cout << "no azure kinect devices detected!" << std::endl;
    }
    // depth mode: K4A_DEPTH_MODE_NFOV_UNBINNED and K4A_DEPTH_MODE_WFOV_2X2BINNED is recommended
    // if depth mode == NFOV, rgb resolution = 4:3 is recommended.
    k4a_device_configuration_t device_config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    device_config.camera_fps = K4A_FRAMES_PER_SECOND_30;
    device_config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
    device_config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    device_config.color_resolution = K4A_COLOR_RESOLUTION_1536P;
    device_config.synchronized_images_only = true;

    std::cout << "Started opening K4A device..." << std::endl;
    m_device = k4a::device::open(K4A_DEVICE_DEFAULT);
    m_device.start_cameras(&device_config);
    std::cout << "Finished opening K4A device." << std::endl;

    k4a::calibration calibration = m_device.get_calibration(device_config.depth_mode, device_config.color_resolution);
    k4a::calibration calibration_downscaled;
    memcpy(&calibration_downscaled, &calibration, sizeof(k4a::calibration));

    DownscaleCalibration(calibration, calibration_downscaled, 3.2);

    m_frame_width_pixels = calibration_downscaled.color_camera_calibration.resolution_width;
    m_frame_height_pixels = calibration_downscaled.color_camera_calibration.resolution_height;

    //print_calibration(calibration);
    //print_calibration(calibration_downscaled);

    m_transformation = k4a::transformation(calibration_downscaled);
    m_cur_frame_num = 0;


    m_save_online_frame = save_online_frame;
    m_data_path = data_path;
}

surfelwarp::AzureKinectDKFetch::~AzureKinectDKFetch()
{
    m_device.close();
}

void surfelwarp::AzureKinectDKFetch::FetchDepthImage(size_t frame_idx, cv::Mat &depth_img)
{
    while (frame_idx >= m_cur_frame_num)
    {
        TakeNewPictureFrame();
    }
    depth_img = m_depth_image_vec[frame_idx];
#ifdef DEBUG
    std::cout << "FetchDepthImage: " << frame_idx << " " << m_cur_frame_num << std::endl;
#endif
}

void surfelwarp::AzureKinectDKFetch::FetchDepthImage(size_t frame_idx, void *depth_img)
{
}

void surfelwarp::AzureKinectDKFetch::FetchRGBImage(size_t frame_idx, cv::Mat &rgb_img)
{
    while (frame_idx >= m_cur_frame_num)
    {
        TakeNewPictureFrame();
    }
    rgb_img = m_color_image_vec[frame_idx];
#ifdef DEBUG
    std::cout << "FetchRGBImage: " << frame_idx << " " << m_cur_frame_num << std::endl;
#endif
}
void surfelwarp::AzureKinectDKFetch::FetchRGBImage(size_t frame_idx, void *rgb_img)
{
}

void surfelwarp::AzureKinectDKFetch::DownscaleCalibration(const k4a::calibration calibration, k4a::calibration &new_calibration, float scale)
{
    new_calibration.color_camera_calibration.resolution_width /= scale;
    new_calibration.color_camera_calibration.resolution_height /= scale;
    new_calibration.color_camera_calibration.intrinsics.parameters.param.cx /= scale;
    new_calibration.color_camera_calibration.intrinsics.parameters.param.cy /= scale;
    new_calibration.color_camera_calibration.intrinsics.parameters.param.fx /= scale;
    new_calibration.color_camera_calibration.intrinsics.parameters.param.fy /= scale;

#ifdef DEBUG
    std::cout << "cx: " << new_calibration.color_camera_calibration.intrinsics.parameters.param.cx << std::endl;
    std::cout << "cy: " << new_calibration.color_camera_calibration.intrinsics.parameters.param.cy << std::endl;
    std::cout << "fx: " << new_calibration.color_camera_calibration.intrinsics.parameters.param.fx << std::endl;
    std::cout << "fy: " << new_calibration.color_camera_calibration.intrinsics.parameters.param.fy << std::endl;
#endif
}

bool surfelwarp::AzureKinectDKFetch::TakeNewPictureFrame()
{
    if (m_device.get_capture(&m_capture, std::chrono::milliseconds(5)))
    {
        m_k4a_depth_image = m_capture.get_depth_image();
        m_k4a_color_image = m_capture.get_color_image();
        m_k4a_transformed_depth_image = m_transformation.depth_image_to_color_camera(m_k4a_depth_image);

        m_depth_image = cv::Mat(m_k4a_transformed_depth_image.get_height_pixels(),
                                m_k4a_transformed_depth_image.get_width_pixels(),
                                CV_16UC1,
                                m_k4a_transformed_depth_image.get_buffer());

        m_color_image_rgba = cv::Mat(m_k4a_color_image.get_height_pixels(),
                                     m_k4a_color_image.get_width_pixels(),
                                     CV_8UC4,
                                     m_k4a_color_image.get_buffer());

        cv::cvtColor(m_color_image_rgba, m_color_image_rgb, cv::COLOR_BGRA2BGR);

        cv::resize(m_color_image_rgb, m_color_image,
                   cv::Size(m_frame_width_pixels, m_frame_height_pixels),
                   0, 0, cv::INTER_AREA);

        m_depth_image_vec.push_back(m_depth_image);
        m_color_image_vec.push_back(m_color_image);

        if (m_save_online_frame)
        {
            std::string file_name = m_data_path.string() +"/frame-";
            char frame_idx_str[20];
            sprintf(frame_idx_str, "%06d", static_cast<int>(m_cur_frame_num));
            file_name += frame_idx_str;
            cv::imwrite(file_name + ".depth.png", m_depth_image);
            cv::imwrite(file_name + ".color.png", m_color_image);
        }

        m_cur_frame_num++;
        return true;
    }
    return false;
}
