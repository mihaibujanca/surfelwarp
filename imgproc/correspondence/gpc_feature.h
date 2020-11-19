#pragma once

#include "imgproc/correspondence/gpc_common.h"
#include "PatchColliderRGBCorrespondence.h"

namespace surfelwarp {
	
	//The declare of feature build
	template<int PatchHalfSize>
	__device__ __forceinline__
	void buildDCTPatchFeature(
		cudaTextureObject_t normalized_rgb, int center_x, int center_y,
		GPCPatchFeature<surfelwarp::PatchColliderRGBCorrespondence::Parameters::feature_dim>& feature
	);
}

#if defined(__CUDACC__)
#include "imgproc/correspondence/gpc_feature.cuh"
#endif