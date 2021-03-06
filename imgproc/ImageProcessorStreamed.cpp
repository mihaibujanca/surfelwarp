#include "imgproc/ImageProcessor.h"
#include <cuda_profiler_api.h>
void surfelwarp::ImageProcessor::initProcessorStream() {
	//Create the stream
	cudaSafeCall(cudaStreamCreate(&m_processor_stream[0]));
	cudaSafeCall(cudaStreamCreate(&m_processor_stream[1]));
//	cudaSafeCall(cudaStreamCreate(&m_processor_stream[2]));
}

void surfelwarp::ImageProcessor::releaseProcessorStream() {
	//Destroy these streams
	cudaSafeCall(cudaStreamDestroy(m_processor_stream[0]));
	cudaSafeCall(cudaStreamDestroy(m_processor_stream[1]));
//	cudaSafeCall(cudaStreamDestroy(m_processor_stream[2]));

	//Assign to null value
	m_processor_stream[0] = 0;
	m_processor_stream[1] = 0;
//	m_processor_stream[2] = 0;
}

void surfelwarp::ImageProcessor::ProcessFrameStreamed(CameraObservation & observation, size_t frame_idx, const cv::Mat* rgb, const cv::Mat* depth) {
    if (rgb && depth) {
        LoadPrevRGBImageFromOpenCV();
        LoadRGBImageFromOpenCV(*rgb);
        LoadDepthImageFromOpenCV(*depth);
    } else {
        FetchFrame(frame_idx);
    }
        UploadDepthImage(m_processor_stream[0]);
	UploadRawRGBImage(m_processor_stream[0]);

	//This seems cause some problem ,disable it at first
	//ReprojectDepthToRGB(stream);

	ClipFilterDepthImage(m_processor_stream[0]); // should come from preprocessor
	ClipNormalizeRGBImage(m_processor_stream[0]);

	//The geometry map
	BuildVertexConfigMap(m_processor_stream[0]);
	BuildNormalRadiusMap(m_processor_stream[0]);
	BuildColorTimeTexture(frame_idx, m_processor_stream[0]);

	//Sync here
	cudaSafeCall(cudaStreamSynchronize(m_processor_stream[0]));

	//Invoke other expensive computations
    // TODO: make sure segmentation on Jetson is bound to m_processor_stream[0]

    SegmentForeground(frame_idx, m_processor_stream[0]); //This doesn't block, even for hashing based method
    ComputeGradientMap(m_processor_stream[0]);
    FindCorrespondence(m_processor_stream[1]); //This will block, thus sync inside

    //The gradient map depends on filtered mask
	//Sync and output
       cudaSafeCall(cudaStreamSynchronize(m_processor_stream[0]));
	memset(&observation, 0, sizeof(observation));

	//The raw depth image for visualization
	observation.raw_depth_img = RawDepthTexture();

	//The geometry maps
	observation.filter_depth_img = FilteredDepthTexture();
	observation.vertex_config_map = VertexConfidTexture();
	observation.normal_radius_map = NormalRadiusTexture();

	//The color maps
	observation.color_time_map = ColorTimeTexture();
	observation.normalized_rgba_map = ClipNormalizedRGBTexture();
	observation.normalized_rgba_prevframe = ClipNormalizedRGBTexturePrev();
	observation.density_map = DensityMapTexture();
	observation.density_gradient_map = DensityGradientTexture();

	//The foreground masks
	observation.foreground_mask = ForegroundMask();
	observation.filter_foreground_mask = FilterForegroundMask();
	observation.foreground_mask_gradient_map = ForegroundMaskGradientTexture();

    cudaSafeCall(cudaStreamSynchronize(m_processor_stream[1])); // this takes time
    //The correspondence pixel pairs
	const auto& pixel_pair_array = CorrespondencePixelPair();
	observation.correspondence_pixel_pairs = DeviceArrayView<ushort4>(pixel_pair_array.ptr(), pixel_pair_array.size());
}