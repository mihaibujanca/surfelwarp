#include "core/warp_solver/WarpSolver.h"

void surfelwarp::WarpSolver::initSolverStream() {
	//Create the stream
	cudaSafeCall(cudaStreamCreate(&m_solver_stream[0]));
	cudaSafeCall(cudaStreamCreate(&m_solver_stream[1]));
	cudaSafeCall(cudaStreamCreate(&m_solver_stream[2]));
	cudaSafeCall(cudaStreamCreate(&m_solver_stream[3]));
	
	//Hand in the stream to pcg solver
	UpdatePCGSolverStream(m_solver_stream[0]);
}

void surfelwarp::WarpSolver::releaseSolverStream() {
	//Update 0 stream to pcg solver
	UpdatePCGSolverStream(0);

	cudaSafeCall(cudaStreamDestroy(m_solver_stream[0]));
	cudaSafeCall(cudaStreamDestroy(m_solver_stream[1]));
	cudaSafeCall(cudaStreamDestroy(m_solver_stream[2]));
	cudaSafeCall(cudaStreamDestroy(m_solver_stream[3]));

	//Assign to null stream
	m_solver_stream[0] = 0;
	m_solver_stream[1] = 0;
	m_solver_stream[2] = 0;
}

void surfelwarp::WarpSolver::syncAllSolverStream() {
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[0]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[1]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[2]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[3]));
}

void surfelwarp::WarpSolver::SolveStreamed() {
	//Actual computation
	buildSolverIndexStreamed();
//	Modifying kNumGaussNewtonIterations has a big impact on computation!
	for(auto i = 0; i < Constants::kNumGaussNewtonIterations; i++) {
        solverIterationStreamed(m_iteration_data.IsGlobalIteration());
	}

//	syncAllSolverStream();
}

void surfelwarp::WarpSolver::buildSolverIndexStreamed() {
	QueryPixelKNN(m_solver_stream[0]); //Sync is required here
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[0]));

	//FetchPotentialDenseImageTermPixelsFixedIndex
	m_image_knn_fetcher->SetInputs(m_knn_map, m_rendered_maps.index_map);
	m_image_knn_fetcher->MarkPotentialMatchedPixels(m_solver_stream[0]);
	m_image_knn_fetcher->CompactPotentialValidPixels(m_solver_stream[0]);
	//m_image_knn_fetcher->SyncQueryCompactedPotentialPixelSize();

	//FindPotentialForegroundMaskPixelSynced();
	setDensityForegroundHandlerFullInput();
	m_density_foreground_handler->MarkValidColorForegroundMaskPixels(m_solver_stream[1]);
	m_density_foreground_handler->CompactValidMaskPixel(m_solver_stream[1]);
	//m_density_foreground_handler->QueryCompactedMaskPixelArraySize();

	//SelectValidSparseFeatureMatchedPairs();
	SetSparseFeatureHandlerFullInput();
	m_sparse_correspondence_handler->ChooseValidPixelPairs(m_solver_stream[2]);
	m_sparse_correspondence_handler->CompactQueryPixelPairs(m_solver_stream[2]);
	//m_sparse_correspondence_handler->QueryCompactedArraySize();

	//The sync group
	m_image_knn_fetcher->SyncQueryCompactedPotentialPixelSize(m_solver_stream[0]); //Sync is inside the method
	m_density_foreground_handler->QueryCompactedMaskPixelArraySize(m_solver_stream[1]); //Sync is inside the method
	m_sparse_correspondence_handler->QueryCompactedArraySize(m_solver_stream[2]); //Sync is inside the method
	setDenseDepthHandlerFullInput();
	setDensityForegroundHandlerFullInput();

	//Before the index part: A sync happened here
	SetNode2TermIndexInput();
	BuildNode2TermIndex(m_solver_stream[0]); //This doesnt block
	BuildNodePair2TermIndexBlocked(m_solver_stream[1]); //This will block

	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[0])); //TODO: remove
}

void surfelwarp::WarpSolver::solverIterationStreamed(bool global) {
	//Hand in the new SE3 to handlers
	m_dense_depth_handler->UpdateNodeSE3(m_iteration_data.CurrentWarpFieldInput());
	m_density_foreground_handler->UpdateNodeSE3(m_iteration_data.CurrentWarpFieldInput());
	m_sparse_correspondence_handler->UpdateNodeSE3(m_iteration_data.CurrentWarpFieldInput());

	//The computation of jacobian
	ComputeTermJacobianIndex(m_solver_stream[0], m_solver_stream[1], m_solver_stream[2], m_solver_stream[3], true); // A sync should happen here
    // Stream 1 and 3 are outputs
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[1]));
    cudaSafeCall(cudaStreamSynchronize(m_solver_stream[3])); // Produces non-crashing error if removed

	//The computation of diagonal blks JtJ and JtError
	SetPreconditionerBuilderAndJtJApplierInput();
	SetJtJMaterializerInput();
    BuildPreconditioner(m_solver_stream[0]);
    if(global) {
        ComputeJtResidualGlobalIteration(m_solver_stream[1]);
        MaterializeJtJNondiagonalBlocksGlobalIteration(m_solver_stream[2]);
    }
	else {
	    ComputeJtResidual(m_solver_stream[1]);
	    MaterializeJtJNondiagonalBlocks(m_solver_stream[2]);
    }
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[2]));


	//The assemble of matrix: a sync here
	const auto diagonal_blks = m_preconditioner_rhs_builder->JtJDiagonalBlocks();
	m_jtj_materializer->AssembleBinBlockCSR(diagonal_blks, m_solver_stream[0]);

	//Debug methods
	//LOG(INFO) << "The total squared residual in materialized, fixed-index solver is " << ComputeTotalResidualSynced(m_solver_stream[0]);

	//Solve it and update
	SolvePCGMaterialized(Constants::kNumPCGMaterializedSolverIterations); // 10 by default. amounts for about 10% of the computation
	m_iteration_data.ApplyWarpFieldUpdate(m_solver_stream[0]);
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[0]));
}