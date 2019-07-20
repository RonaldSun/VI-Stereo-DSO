/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/FullSystem.h"
  
#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ResidualProjections.h"
#include "FullSystem/ImmaturePoint.h"

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/CoarseInitializer.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "util/ImageAndExposure.h"

#include <cmath>

namespace dso
{
int FrameHessian::instanceCounter=0;
int PointHessian::instanceCounter=0;
int CalibHessian::instanceCounter=0;

void quaternionNormalize(Eigen::Vector4d& q) {
  double norm = q.norm();
  q = q / norm;
  return;
}

Eigen::Vector4d rotationToQuaternion(const Eigen::Matrix3d& R) {

    Eigen::Vector4d q = Eigen::Vector4d::Zero();
    double trR = R(0,0)+R(1,1)+R(2,2);
    q(3) = sqrt(trR+1)/2;
    q(0) = (R(2,1) - R(1,2))/4/q(3);
    q(1) = (R(0,2) - R(2,0))/4/q(3);
    q(2) = (R(1,0) - R(0,1))/4/q(3);
    quaternionNormalize(q);
    return q;
}

FullSystem::FullSystem()
{

	int retstat =0;
	if(setting_logStuff)
	{

		retstat += system("rm -rf logs");
		retstat += system("mkdir logs");

		retstat += system("rm -rf mats");
		retstat += system("mkdir mats");

		calibLog = new std::ofstream();
		calibLog->open("logs/calibLog.txt", std::ios::trunc | std::ios::out);
		calibLog->precision(12);

		numsLog = new std::ofstream();
		numsLog->open("logs/numsLog.txt", std::ios::trunc | std::ios::out);
		numsLog->precision(10);

		coarseTrackingLog = new std::ofstream();
		coarseTrackingLog->open("logs/coarseTrackingLog.txt", std::ios::trunc | std::ios::out);
		coarseTrackingLog->precision(10);

		eigenAllLog = new std::ofstream();
		eigenAllLog->open("logs/eigenAllLog.txt", std::ios::trunc | std::ios::out);
		eigenAllLog->precision(10);

		eigenPLog = new std::ofstream();
		eigenPLog->open("logs/eigenPLog.txt", std::ios::trunc | std::ios::out);
		eigenPLog->precision(10);

		eigenALog = new std::ofstream();
		eigenALog->open("logs/eigenALog.txt", std::ios::trunc | std::ios::out);
		eigenALog->precision(10);

		DiagonalLog = new std::ofstream();
		DiagonalLog->open("logs/diagonal.txt", std::ios::trunc | std::ios::out);
		DiagonalLog->precision(10);

		variancesLog = new std::ofstream();
		variancesLog->open("logs/variancesLog.txt", std::ios::trunc | std::ios::out);
		variancesLog->precision(10);


		nullspacesLog = new std::ofstream();
		nullspacesLog->open("logs/nullspacesLog.txt", std::ios::trunc | std::ios::out);
		nullspacesLog->precision(10);
	}
	else
	{
		nullspacesLog=0;
		variancesLog=0;
		DiagonalLog=0;
		eigenALog=0;
		eigenPLog=0;
		eigenAllLog=0;
		numsLog=0;
		calibLog=0;
	}

	assert(retstat!=293847);



	selectionMap = new float[wG[0]*hG[0]];

	coarseDistanceMap = new CoarseDistanceMap(wG[0], hG[0]);
	coarseTracker = new CoarseTracker(wG[0], hG[0]);
	coarseTracker_forNewKF = new CoarseTracker(wG[0], hG[0]);
	coarseInitializer = new CoarseInitializer(wG[0], hG[0]);
	pixelSelector = new PixelSelector(wG[0], hG[0]);

	statistics_lastNumOptIts=0;
	statistics_numDroppedPoints=0;
	statistics_numActivatedPoints=0;
	statistics_numCreatedPoints=0;
	statistics_numForceDroppedResBwd = 0;
	statistics_numForceDroppedResFwd = 0;
	statistics_numMargResFwd = 0;
	statistics_numMargResBwd = 0;

	lastCoarseRMSE.setConstant(100);

	currentMinActDist=2;
	initialized=false;


	ef = new EnergyFunctional();
	ef->red = &this->treadReduce;

	isLost=false;
	initFailed=false;


	needNewKFAfter = -1;

	linearizeOperation=true;
	runMapping=true;
	mappingThread = boost::thread(&FullSystem::mappingLoop, this);
	lastRefStopID=0;



	minIdJetVisDebug = -1;
	maxIdJetVisDebug = -1;
	minIdJetVisTracker = -1;
	maxIdJetVisTracker = -1;
}

FullSystem::~FullSystem()
{
	blockUntilMappingIsFinished();

	if(setting_logStuff)
	{
		calibLog->close(); delete calibLog;
		numsLog->close(); delete numsLog;
		coarseTrackingLog->close(); delete coarseTrackingLog;
		//errorsLog->close(); delete errorsLog;
		eigenAllLog->close(); delete eigenAllLog;
		eigenPLog->close(); delete eigenPLog;
		eigenALog->close(); delete eigenALog;
		DiagonalLog->close(); delete DiagonalLog;
		variancesLog->close(); delete variancesLog;
		nullspacesLog->close(); delete nullspacesLog;
	}

	delete[] selectionMap;

	for(FrameShell* s : allFrameHistory)
		delete s;
	for(FrameHessian* fh : unmappedTrackedFrames)
		delete fh;

	delete coarseDistanceMap;
	delete coarseTracker;
	delete coarseTracker_forNewKF;
	delete coarseInitializer;
	delete pixelSelector;
	delete ef;
}

void FullSystem::setOriginalCalib(const VecXf &originalCalib, int originalW, int originalH)
{

}

void FullSystem::setGammaFunction(float* BInv)
{
	if(BInv==0) return;

	// copy BInv.
	memcpy(Hcalib.Binv, BInv, sizeof(float)*256);


	// invert.
	for(int i=1;i<255;i++)
	{
		// find val, such that Binv[val] = i.
		// I dont care about speed for this, so do it the stupid way.

		for(int s=1;s<255;s++)
		{
			if(BInv[s] <= i && BInv[s+1] >= i)
			{
				Hcalib.B[i] = s+(i - BInv[s]) / (BInv[s+1]-BInv[s]);
				break;
			}
		}
	}
	Hcalib.B[0] = 0;
	Hcalib.B[255] = 255;
}



void FullSystem::printResult(std::string file)
{
	boost::unique_lock<boost::mutex> lock(trackMutex);
	boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

	std::ofstream myfile;
	myfile.open (file.c_str());
	myfile << std::setprecision(15);

	for(FrameShell* s : allFrameHistory)
	{
		if(!s->poseValid) continue;

		if(setting_onlyLogKFPoses && s->marginalizedAt == s->id) continue;

		myfile << s->timestamp <<
			" " << s->camToWorld.translation().transpose()<<
			" " << s->camToWorld.so3().unit_quaternion().x()<<
			" " << s->camToWorld.so3().unit_quaternion().y()<<
			" " << s->camToWorld.so3().unit_quaternion().z()<<
			" " << s->camToWorld.so3().unit_quaternion().w() << "\n";
	}
	myfile.close();
}


Vec4 FullSystem::trackNewCoarse(FrameHessian* fh)
{

	assert(allFrameHistory.size() > 0);
	// set pose initialization.

    for(IOWrap::Output3DWrapper* ow : outputWrapper)
        ow->pushLiveFrame(fh);



	FrameHessian* lastF = coarseTracker->lastRef;

	AffLight aff_last_2_l = AffLight(0,0);

	std::vector<SE3,Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;
	if(use_stereo&&(allFrameHistory.size() == 2||first_track_flag==false)/*frameHessians.size()==1*/){
// 		initializeFromInitializer(fh);
		first_track_flag = true;
		lastF_2_fh_tries.push_back(SE3(Eigen::Matrix<double, 3, 3>::Identity(), Eigen::Matrix<double,3,1>::Zero() ));
		
		for(float rotDelta=0.02; rotDelta < 0.05; rotDelta = rotDelta + 0.02)
		{
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,0,rotDelta), Vec3(0,0,0)));			// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,-rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,0,-rotDelta), Vec3(0,0,0)));			// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		}
		
		coarseTracker->makeK(&Hcalib);
		coarseTracker->setCTRefForFirstFrame(frameHessians);
		lastF = coarseTracker->lastRef;
	}
	else if(allFrameHistory.size() == 2){
		for(unsigned int i=0;i<lastF_2_fh_tries.size();i++) lastF_2_fh_tries.push_back(SE3());
	}
	else
	{
		FrameShell* slast = allFrameHistory[allFrameHistory.size()-2];
		FrameShell* sprelast = allFrameHistory[allFrameHistory.size()-3];
		SE3 slast_2_sprelast;
		SE3 lastF_2_slast;
		{	// lock on global pose consistency!
			boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
			slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;
			lastF_2_slast = slast->camToWorld.inverse() * lastF->shell->camToWorld;
			aff_last_2_l = slast->aff_g2l;
		}
		SE3 fh_2_slast = slast_2_sprelast;// assumed to be the same as fh_2_slast.


		// get last delta-movement.
		lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast);	// assume constant motion.
		lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);	// assume double motion (frame skipped)
		lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*0.5).inverse() * lastF_2_slast); // assume half motion.
		lastF_2_fh_tries.push_back(lastF_2_slast); // assume zero motion.
		lastF_2_fh_tries.push_back(SE3()); // assume zero motion FROM KF.


		// just try a TON of different initializations (all rotations). In the end,
		// if they don't work they will only be tried on the coarsest level, which is super fast anyway.
		// also, if tracking rails here we loose, so we really, really want to avoid that.
		for(float rotDelta=0.02; rotDelta < 0.05; rotDelta++)
		{
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,rotDelta), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,-rotDelta), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		}

		if(!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid)
		{
			lastF_2_fh_tries.clear();
			lastF_2_fh_tries.push_back(SE3());
		}
	}

	Vec3 flowVecs = Vec3(100,100,100);
	SE3 lastF_2_fh = SE3();
	AffLight aff_g2l = AffLight(0,0);


	// as long as maxResForImmediateAccept is not reached, I'll continue through the options.
	// I'll keep track of the so-far best achieved residual for each level in achievedRes.
	// If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.


	Vec5 achievedRes = Vec5::Constant(NAN);
	bool haveOneGood = false;
	int tryIterations=0;
	for(unsigned int i=0;i<lastF_2_fh_tries.size();i++)
	{
		if(use_stereo&&frameHessians.size()<setting_maxFrames-1){
		      if(i>0){
			initFailed = true;
			first_track_flag = false;
		      }

		  }
		AffLight aff_g2l_this = aff_last_2_l;
		SE3 lastF_2_fh_this = lastF_2_fh_tries[i];
		bool trackingIsGood = coarseTracker->trackNewestCoarse(
				fh, lastF_2_fh_this, aff_g2l_this,
				pyrLevelsUsed-1,
				achievedRes);	// in each level has to be at least as good as the last try.
		tryIterations++;
// 		LOG(INFO)<<"coarseTracker->lastResiduals[0]: "<<coarseTracker->lastResiduals[0];
		if(i != 0)
		{
			printf("RE-TRACK ATTEMPT %d with initOption %d and start-lvl %d (ab %f %f): %f %f %f %f %f -> %f %f %f %f %f \n",
					i,
					i, pyrLevelsUsed-1,
					aff_g2l_this.a,aff_g2l_this.b,
					achievedRes[0],
					achievedRes[1],
					achievedRes[2],
					achievedRes[3],
					achievedRes[4],
					coarseTracker->lastResiduals[0],
					coarseTracker->lastResiduals[1],
					coarseTracker->lastResiduals[2],
					coarseTracker->lastResiduals[3],
					coarseTracker->lastResiduals[4]);
		}
// 		if(i>10&&frameHessians.size()==1){
// 		  initFailed = true;
// 		}

		// do we have a new winner?
		if(trackingIsGood && std::isfinite((float)coarseTracker->lastResiduals[0]) && !(coarseTracker->lastResiduals[0] >=  achievedRes[0]))
		{
			//printf("take over. minRes %f -> %f!\n", achievedRes[0], coarseTracker->lastResiduals[0]);
			flowVecs = coarseTracker->lastFlowIndicators;
			aff_g2l = aff_g2l_this;
			lastF_2_fh = lastF_2_fh_this;
			haveOneGood = true;
		}

		// take over achieved res (always).
		if(haveOneGood)
		{
			for(int i=0;i<5;i++)
			{
				if(!std::isfinite((float)achievedRes[i]) || achievedRes[i] > coarseTracker->lastResiduals[i])	// take over if achievedRes is either bigger or NAN.
					achievedRes[i] = coarseTracker->lastResiduals[i];
			}
		}


		if(haveOneGood &&  achievedRes[0] < lastCoarseRMSE[0]*setting_reTrackThreshold)
		    break;

	}
// 	if(frameHessians.size()==1){
// 	    if(!haveOneGood){
// 	      initFailed = true;
// 	    }
// 	    LOG(INFO)<<"achievedRes[0]: "<<achievedRes[0];
// 	    if(achievedRes[0]>11){
// 	      initFailed = true;
// 	    }
// 	}
	if(!haveOneGood)
	{
        printf("BIG ERROR! tracking failed entirely. Take predictred pose and hope we may somehow recover.\n");
		flowVecs = Vec3(0,0,0);
		aff_g2l = aff_last_2_l;
		lastF_2_fh = lastF_2_fh_tries[0];
	}

	lastCoarseRMSE = achievedRes;

	// no lock required, as fh is not used anywhere yet.
	fh->shell->camToTrackingRef = lastF_2_fh.inverse();
	fh->shell->trackingRef = lastF->shell;
	fh->shell->aff_g2l = aff_g2l;
	fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;


	if(coarseTracker->firstCoarseRMSE < 0)
		coarseTracker->firstCoarseRMSE = achievedRes[0];

    if(!setting_debugout_runquiet)
        printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n", aff_g2l.a, aff_g2l.b, fh->ab_exposure, achievedRes[0]);



	if(setting_logStuff)
	{
		(*coarseTrackingLog) << std::setprecision(16)
						<< fh->shell->id << " "
						<< fh->shell->timestamp << " "
						<< fh->ab_exposure << " "
						<< fh->shell->camToWorld.log().transpose() << " "
						<< aff_g2l.a << " "
						<< aff_g2l.b << " "
						<< achievedRes[0] << " "
						<< tryIterations << "\n";
	}


	return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
}

void FullSystem::traceNewCoarse(FrameHessian* fh)
{
	boost::unique_lock<boost::mutex> lock(mapMutex);

	int trace_total=0, trace_good=0, trace_oob=0, trace_out=0, trace_skip=0, trace_badcondition=0, trace_uninitialized=0;

	Mat33f K = Mat33f::Identity();
	K(0,0) = Hcalib.fxl();
	K(1,1) = Hcalib.fyl();
	K(0,2) = Hcalib.cxl();
	K(1,2) = Hcalib.cyl();

	for(FrameHessian* host : frameHessians)		// go through all active frames
	{

		SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
		Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
		Vec3f Kt = K * hostToNew.translation().cast<float>();

		Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(), fh->aff_g2l()).cast<float>();

		for(ImmaturePoint* ph : host->immaturePoints)
		{
			ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false );

			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_GOOD) trace_good++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OOB) trace_oob++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OUTLIER) trace_out++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
			trace_total++;
		}
	}
//	printf("ADD: TRACE: %'d points. %'d (%.0f%%) good. %'d (%.0f%%) skip. %'d (%.0f%%) badcond. %'d (%.0f%%) oob. %'d (%.0f%%) out. %'d (%.0f%%) uninit.\n",
//			trace_total,
//			trace_good, 100*trace_good/(float)trace_total,
//			trace_skip, 100*trace_skip/(float)trace_total,
//			trace_badcondition, 100*trace_badcondition/(float)trace_total,
//			trace_oob, 100*trace_oob/(float)trace_total,
//			trace_out, 100*trace_out/(float)trace_total,
//			trace_uninitialized, 100*trace_uninitialized/(float)trace_total);
}

// process nonkey frame to refine key frame idepth
void FullSystem::traceNewCoarseNonKey(FrameHessian *fh, FrameHessian *fh_right) {
	boost::unique_lock<boost::mutex> lock(mapMutex);
	// new idepth after refinement
	float idepth_min_update = 0;
	float idepth_max_update = 0;

	Mat33f K = Mat33f::Identity();
	K(0, 0) = Hcalib.fxl();
	K(1, 1) = Hcalib.fyl();
	K(0, 2) = Hcalib.cxl();
	K(1, 2) = Hcalib.cyl();

	Mat33f Ki = K.inverse();


	for (FrameHessian *host : frameHessians)        // go through all active frames
	{
//		number++;
		int trace_total = 0, trace_good = 0, trace_oob = 0, trace_out = 0, trace_skip = 0, trace_badcondition = 0, trace_uninitialized = 0;

		// trans from reference keyframe to newest frame
		SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
		// KRK-1
		Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
		// KRi
		Mat33f KRi = K * hostToNew.rotationMatrix().inverse().cast<float>();
		// Kt
		Vec3f Kt = K * hostToNew.translation().cast<float>();
		// t
		Vec3f t = hostToNew.translation().cast<float>();

		//aff
		Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(), fh->aff_g2l()).cast<float>();

		for (ImmaturePoint *ph : host->immaturePoints)
		{
			// do temperol stereo match
			ImmaturePointStatus phTrackStatus = ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false);
			if (phTrackStatus == ImmaturePointStatus::IPS_GOOD)
			{
				ImmaturePoint *phNonKey = new ImmaturePoint(ph->lastTraceUV(0), ph->lastTraceUV(1), fh, &Hcalib);

				// project onto newest frame
				Vec3f ptpMin = KRKi * (Vec3f(ph->u, ph->v, 1) / ph->idepth_min) + Kt;
				float idepth_min_project = 1.0f / ptpMin[2];
				Vec3f ptpMax = KRKi * (Vec3f(ph->u, ph->v, 1) / ph->idepth_max) + Kt;
				float idepth_max_project = 1.0f / ptpMax[2];

				phNonKey->idepth_min = idepth_min_project;
				phNonKey->idepth_max = idepth_max_project;
				phNonKey->u_stereo = phNonKey->u;
				phNonKey->v_stereo = phNonKey->v;
				phNonKey->idepth_min_stereo = phNonKey->idepth_min;
				phNonKey->idepth_max_stereo = phNonKey->idepth_max;

				// do static stereo match from left image to right
				ImmaturePointStatus phNonKeyStereoStatus = phNonKey->traceStereo(fh_right, &Hcalib, 1);
				if(phNonKeyStereoStatus == ImmaturePointStatus::IPS_GOOD)
				{
				    ImmaturePoint* phNonKeyRight = new ImmaturePoint(phNonKey->lastTraceUV(0), phNonKey->lastTraceUV(1), fh_right, &Hcalib );

				    phNonKeyRight->u_stereo = phNonKeyRight->u;
				    phNonKeyRight->v_stereo = phNonKeyRight->v;
				    phNonKeyRight->idepth_min_stereo = phNonKey->idepth_min;
				    phNonKeyRight->idepth_max_stereo = phNonKey->idepth_max;

				    // do static stereo match from right image to left
				    ImmaturePointStatus  phNonKeyRightStereoStatus = phNonKeyRight->traceStereo(fh, &Hcalib, 0);

				    // change of u after two different stereo match
				    float u_stereo_delta = abs(phNonKey->u_stereo - phNonKeyRight->lastTraceUV(0));
				    float disparity = phNonKey->u_stereo - phNonKey->lastTraceUV[0];

				    // free to debug the threshold
				    if(u_stereo_delta > 1 && disparity < 10)
				    {
					ph->lastTraceStatus = ImmaturePointStatus :: IPS_OUTLIER;
					continue;
				    }
				    else
				    {
					// project back
					Vec3f pinverse_min = KRi * (Ki * Vec3f(phNonKey->u_stereo, phNonKey->v_stereo, 1) / phNonKey->idepth_min_stereo - t);
					idepth_min_update = 1.0f / pinverse_min(2);

					Vec3f pinverse_max = KRi * (Ki * Vec3f(phNonKey->u_stereo, phNonKey->v_stereo, 1) / phNonKey->idepth_max_stereo - t);
					idepth_max_update = 1.0f / pinverse_max(2);

					ph->idepth_min = idepth_min_update;
					ph->idepth_max = idepth_max_update;

					delete phNonKey;
					delete phNonKeyRight;
				    }
				}
				else
				{
				    delete phNonKey;
				    continue;
				}
			}
			if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD) trace_good++;
			if (ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
			if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OOB) trace_oob++;
			if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER) trace_out++;
			if (ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
			if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
			trace_total++;
		}
	
	}
}


//process keyframe
void FullSystem::traceNewCoarseKey(FrameHessian* fh, FrameHessian* fh_right)
{
		boost::unique_lock<boost::mutex> lock(mapMutex);

		int trace_total=0, trace_good=0, trace_oob=0, trace_out=0, trace_skip=0, trace_badcondition=0, trace_uninitialized=0;

		Mat33f K = Mat33f::Identity();
		K(0,0) = Hcalib.fxl();
		K(1,1) = Hcalib.fyl();
		K(0,2) = Hcalib.cxl();
		K(1,2) = Hcalib.cyl();

		for(FrameHessian* host : frameHessians)		// go through all active frames
		{

			// trans from reference key frame to the newest one
			SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
			//KRK-1
			Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
			//Kt
			Vec3f Kt = K * hostToNew.translation().cast<float>();

			Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(), fh->aff_g2l()).cast<float>();

			for(ImmaturePoint* ph : host->immaturePoints)
			{
				ImmaturePointStatus phTrackStatus = ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false );
// 				if(ph->u==145&&ph->v==21){
// 				    LOG(INFO)<<"trace depthmax: "<<ph->idepth_max<<" min: "<<ph->idepth_min;
// 				}
				if(ph->lastTraceStatus==ImmaturePointStatus::IPS_GOOD) trace_good++;
				if(ph->lastTraceStatus==ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
				if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OOB) trace_oob++;
				if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OUTLIER) trace_out++;
				if(ph->lastTraceStatus==ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
				if(ph->lastTraceStatus==ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
				trace_total++;
			}
		}

}


void FullSystem::activatePointsMT_Reductor(
		std::vector<PointHessian*>* optimized,
		std::vector<ImmaturePoint*>* toOptimize,
		int min, int max, Vec10* stats, int tid)
{
	ImmaturePointTemporaryResidual* tr = new ImmaturePointTemporaryResidual[frameHessians.size()];
	for(int k=min;k<max;k++)
	{
		(*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k],1,tr);
	}
	delete[] tr;
}



void FullSystem::activatePointsMT()
{
	if(ef->nPoints < setting_desiredPointDensity*0.66)
		currentMinActDist -= 0.8;
	if(ef->nPoints < setting_desiredPointDensity*0.8)
		currentMinActDist -= 0.5;
	else if(ef->nPoints < setting_desiredPointDensity*0.9)
		currentMinActDist -= 0.2;
	else if(ef->nPoints < setting_desiredPointDensity)
		currentMinActDist -= 0.1;

	if(ef->nPoints > setting_desiredPointDensity*1.5)
		currentMinActDist += 0.8;
	if(ef->nPoints > setting_desiredPointDensity*1.3)
		currentMinActDist += 0.5;
	if(ef->nPoints > setting_desiredPointDensity*1.15)
		currentMinActDist += 0.2;
	if(ef->nPoints > setting_desiredPointDensity)
		currentMinActDist += 0.1;

	if(currentMinActDist < 0) currentMinActDist = 0;
	if(currentMinActDist > 4) currentMinActDist = 4;

    if(!setting_debugout_runquiet)
        printf("SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
                currentMinActDist, (int)(setting_desiredPointDensity), ef->nPoints);



	FrameHessian* newestHs = frameHessians.back();

	// make dist map.
	coarseDistanceMap->makeK(&Hcalib);
	coarseDistanceMap->makeDistanceMap(frameHessians, newestHs);

	//coarseTracker->debugPlotDistMap("distMap");

	std::vector<ImmaturePoint*> toOptimize; toOptimize.reserve(20000);

	for(FrameHessian* host : frameHessians)		// go through all active frames
	{
		if(host == newestHs) continue;

		SE3 fhToNew = newestHs->PRE_worldToCam * host->PRE_camToWorld;
		Mat33f KRKi = (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() * coarseDistanceMap->Ki[0]);
		Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());


		for(unsigned int i=0;i<host->immaturePoints.size();i+=1)
		{
			ImmaturePoint* ph = host->immaturePoints[i];
			ph->idxInImmaturePoints = i;

			// delete points that have never been traced successfully, or that are outlier on the last trace.
			if(!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER)
			{
//				immature_invalid_deleted++;
				// remove point.
				delete ph;
				host->immaturePoints[i]=0;
				continue;
			}

			// can activate only if this is true.
			bool canActivate = (ph->lastTraceStatus == IPS_GOOD
					|| ph->lastTraceStatus == IPS_SKIPPED
					|| ph->lastTraceStatus == IPS_BADCONDITION
					|| ph->lastTraceStatus == IPS_OOB )
							&& ph->lastTracePixelInterval < 8
							&& ph->quality > setting_minTraceQuality
							&& (ph->idepth_max+ph->idepth_min) > 0;


			// if I cannot activate the point, skip it. Maybe also delete it.
			if(!canActivate)
			{
				// if point will be out afterwards, delete it instead.
				if(ph->host->flaggedForMarginalization || ph->lastTraceStatus == IPS_OOB)
				{
//					immature_notReady_deleted++;
					delete ph;
					host->immaturePoints[i]=0;
				}
//				immature_notReady_skipped++;
				continue;
			}


			// see if we need to activate point due to distance map.
			Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt*(0.5f*(ph->idepth_max+ph->idepth_min));
			int u = ptp[0] / ptp[2] + 0.5f;
			int v = ptp[1] / ptp[2] + 0.5f;

			if((u > 0 && v > 0 && u < wG[1] && v < hG[1]))
			{

				float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u+wG[1]*v] + (ptp[0]-floorf((float)(ptp[0])));

				if(dist>=currentMinActDist* ph->my_type)
				{
					coarseDistanceMap->addIntoDistFinal(u,v);
					toOptimize.push_back(ph);
				}
			}
			else
			{
				delete ph;
				host->immaturePoints[i]=0;
			}
		}
	}
// 	LOG(INFO)<<"toOptimize.size(): "<<toOptimize.size();
//	printf("ACTIVATE: %d. (del %d, notReady %d, marg %d, good %d, marg-skip %d)\n",
//			(int)toOptimize.size(), immature_deleted, immature_notReady, immature_needMarg, immature_want, immature_margskip);

	std::vector<PointHessian*> optimized; optimized.resize(toOptimize.size());

	if(multiThreading)
		treadReduce.reduce(boost::bind(&FullSystem::activatePointsMT_Reductor, this, &optimized, &toOptimize, _1, _2, _3, _4), 0, toOptimize.size(), 50);

	else
		activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);

// 	LOG(INFO)<<"toOptimize.size(): "<<toOptimize.size();
	for(unsigned k=0;k<toOptimize.size();k++)
	{
		PointHessian* newpoint = optimized[k];
		ImmaturePoint* ph = toOptimize[k];

		if(newpoint != 0 && newpoint != (PointHessian*)((long)(-1)))
		{
			newpoint->host->immaturePoints[ph->idxInImmaturePoints]=0;
			newpoint->host->pointHessians.push_back(newpoint);
			ef->insertPoint(newpoint);
			for(PointFrameResidual* r : newpoint->residuals)
				ef->insertResidual(r);
			assert(newpoint->efPoint != 0);
			delete ph;
		}
		else if(newpoint == (PointHessian*)((long)(-1)) || ph->lastTraceStatus==IPS_OOB)
		{
			delete ph;
			ph->host->immaturePoints[ph->idxInImmaturePoints]=0;
		}
		else
		{
			assert(newpoint == 0 || newpoint == (PointHessian*)((long)(-1)));
		}
	}


	for(FrameHessian* host : frameHessians)
	{
		for(int i=0;i<(int)host->immaturePoints.size();i++)
		{
			if(host->immaturePoints[i]==0)
			{
				host->immaturePoints[i] = host->immaturePoints.back();
				host->immaturePoints.pop_back();
				i--;
			}
		}
	}


}






void FullSystem::activatePointsOldFirst()
{
	assert(false);
}

void FullSystem::flagPointsForRemoval()
{
	assert(EFIndicesValid);

	std::vector<FrameHessian*> fhsToKeepPoints;
	std::vector<FrameHessian*> fhsToMargPoints;

	//if(setting_margPointVisWindow>0)
	{
		for(int i=((int)frameHessians.size())-1;i>=0 && i >= ((int)frameHessians.size());i--)
			if(!frameHessians[i]->flaggedForMarginalization) fhsToKeepPoints.push_back(frameHessians[i]);

		for(int i=0; i< (int)frameHessians.size();i++)
			if(frameHessians[i]->flaggedForMarginalization) fhsToMargPoints.push_back(frameHessians[i]);
	}



	//ef->setAdjointsF();
	//ef->setDeltaF(&Hcalib);
	int flag_oob=0, flag_in=0, flag_inin=0, flag_nores=0;

	for(FrameHessian* host : frameHessians)		// go through all active frames
	{
		for(unsigned int i=0;i<host->pointHessians.size();i++)
		{
			PointHessian* ph = host->pointHessians[i];
			if(ph==0) continue;

			if(ph->idepth_scaled < 0 || ph->residuals.size()==0)
			{
				host->pointHessiansOut.push_back(ph);
				ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
				host->pointHessians[i]=0;
				flag_nores++;
			}
			else if(ph->isOOB(fhsToKeepPoints, fhsToMargPoints) || host->flaggedForMarginalization)
			{
				flag_oob++;
				if(ph->isInlierNew())
				{
					flag_in++;
					int ngoodRes=0;
					for(PointFrameResidual* r : ph->residuals)
					{
// 						if(r->efResidual->idxInAll==0)continue;
						r->resetOOB();
						if(r->stereoResidualFlag == true)
							r->linearizeStereo(&Hcalib);
						else
							r->linearize(&Hcalib);
// 						r->linearize(&Hcalib);
						r->efResidual->isLinearized = false;
						r->applyRes(true);
						if(r->efResidual->isActive())
						{
							r->efResidual->fixLinearizationF(ef);
							ngoodRes++;
						}
					}
					if(ph->idepth_hessian > setting_minIdepthH_marg)
					{
						flag_inin++;
						ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
						host->pointHessiansMarginalized.push_back(ph);
					}
					else
					{
						ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
						host->pointHessiansOut.push_back(ph);
					}


				}
				else
				{
					host->pointHessiansOut.push_back(ph);
					ph->efPoint->stateFlag = EFPointStatus::PS_DROP;


					//printf("drop point in frame %d (%d goodRes, %d activeRes)\n", ph->host->idx, ph->numGoodResiduals, (int)ph->residuals.size());
				}

				host->pointHessians[i]=0;
			}
		}


		for(int i=0;i<(int)host->pointHessians.size();i++)
		{
			if(host->pointHessians[i]==0)
			{
				host->pointHessians[i] = host->pointHessians.back();
				host->pointHessians.pop_back();
				i--;
			}
		}
	}

}


void FullSystem::addActiveFrame( ImageAndExposure* image, ImageAndExposure* image_right, int id )
{
	LOG(INFO)<<"id: "<<id;
	LOG(INFO)<<"M_num: "<<M_num<<" M_num2: "<<M_num2;
	run_time = pic_time_stamp[id];
	LOG(INFO)<<std::fixed<<std::setprecision(12)<<"run_time: "<<run_time;
	if(isLost) return;
	boost::unique_lock<boost::mutex> lock(trackMutex);
	
	if(use_stereo&&(T_WD.scale()>2||T_WD.scale()<0.6)){
	    initFailed = true;
	    first_track_flag = false;
	}
// 	LOG(INFO)<<"allKeyFramesHistory.size(): "<<allKeyFramesHistory.size();
	if(use_stereo==false&&(T_WD.scale()<0.1||T_WD.scale()>10)){
	    initFailed = true;
	    first_track_flag = false;
	}
// 	if(use_stereo==false){
// 	    if(allKeyFramesHistory.size()<40){
// 		imu_use_flag = false;
// 	    }else{
// 		imu_use_flag = true;
// 	    }
// 	}
	// =========================== add into allFrameHistory =========================
	FrameHessian* fh = new FrameHessian();
	FrameHessian* fh_right = new FrameHessian();
	FrameShell* shell = new FrameShell();
	shell->camToWorld = SE3(); 		// no lock required, as fh is not used anywhere yet.
	shell->aff_g2l = AffLight(0,0);
	shell->marginalizedAt = shell->id = allFrameHistory.size();
	shell->timestamp = image->timestamp;
	shell->incoming_id = id;
	fh->shell = shell;
	fh_right->shell=shell;


	// =========================== make Images / derivatives etc. =========================
	fh->ab_exposure = image->exposure_time;
	fh->makeImages(image->image, &Hcalib);
	fh_right->ab_exposure = image_right->exposure_time;
	fh_right->makeImages(image_right->image,&Hcalib);
	fh->frame_right = fh_right;
	
	if(allFrameHistory.size()>0){
	    fh->velocity = fh->shell->velocity = allFrameHistory.back()->velocity;
	    fh->bias_g = fh->shell->bias_g = allFrameHistory.back()->bias_g + allFrameHistory.back()->delta_bias_g;
	    fh->bias_a = fh->shell->bias_a = allFrameHistory.back()->bias_a + allFrameHistory.back()->delta_bias_a;
	}
	
	allFrameHistory.push_back(shell);
	
// 	LOG(INFO)<<"fh->bias_a: "<<fh->bias_a.transpose();
	if(!initialized)
	{
		// use initializer!
		if(coarseInitializer->frameID<0&&use_stereo)	// first frame set. fh is kept by coarseInitializer.
		{
			coarseInitializer->setFirstStereo(&Hcalib, fh,fh_right);
// 			coarseInitializer->setFirst(&Hcalib, fh);
			initFirstFrame_imu(fh);
			
			initializeFromInitializer(fh);
			initialized = true;
			M_num = 0;
			M_num2 = 0;

		}
		else if(coarseInitializer->frameID<0){
			coarseInitializer->setFirst(&Hcalib, fh);
			initFirstFrame_imu(fh);
		}
		else if(coarseInitializer->trackFrame(fh, outputWrapper))	// if SNAPPED
		{
			initializeFromInitializer(fh);
			lock.unlock();
			deliverTrackedFrame(fh, fh_right, true);
			
		}
		else
		{
			// if still initializing
			fh->shell->poseValid = false;
			delete fh;
		}
		std::ofstream f1;
		std::string dsoposefile = "./data/"+savefile_tail+".txt";
		f1.open(dsoposefile,std::ios::out);
		f1.close();
		std::ofstream f2;
		std::string gtfile = "./data/"+savefile_tail+"_gt.txt";
		f2.open(gtfile,std::ios::out);
		f2.close();
		return;
	}
	else	// do front-end operation.
	{
		// =========================== SWAP tracking reference?. =========================
		if(coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID)
		{
			boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
			CoarseTracker* tmp = coarseTracker; 
			coarseTracker=coarseTracker_forNewKF; 
			coarseTracker_forNewKF=tmp;
		}
		
// 		if(allFrameHistory.size() == 2){
// 			initializeFromInitializer(fh);
// 		}
		Vec4 tres = trackNewCoarse(fh);
// 		LOG(INFO)<<"track done";
		
		if(!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) || !std::isfinite((double)tres[2]) || !std::isfinite((double)tres[3]))
		{
		    printf("Initial Tracking failed: LOST!\n");
				isLost=true;
		    return;
		}

		bool needToMakeKF = false;
		if(setting_keyframesPerSecond > 0)
		{
			needToMakeKF = allFrameHistory.size()== 1 ||
					(fh->shell->timestamp - allKeyFramesHistory.back()->timestamp) > 0.95f/setting_keyframesPerSecond;
		}
		else
		{
			Vec2 refToFh=AffLight::fromToVecExposure(coarseTracker->lastRef->ab_exposure, fh->ab_exposure,
					coarseTracker->lastRef_aff_g2l, fh->shell->aff_g2l);

			// BRIGHTNESS CHECK
			needToMakeKF = allFrameHistory.size()== 1 ||
					setting_kfGlobalWeight*setting_maxShiftWeightT *  sqrtf((double)tres[1]) / (wG[0]+hG[0]) +
					setting_kfGlobalWeight*setting_maxShiftWeightR *  sqrtf((double)tres[2]) / (wG[0]+hG[0]) +
					setting_kfGlobalWeight*setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (wG[0]+hG[0]) +
					setting_kfGlobalWeight*setting_maxAffineWeight * fabs(logf((float)refToFh[0])) > 1 ||
					2*coarseTracker->firstCoarseRMSE < tres[0];
			double delta = setting_kfGlobalWeight*setting_maxShiftWeightT *  sqrtf((double)tres[1]) / (wG[0]+hG[0]) +
					setting_kfGlobalWeight*setting_maxShiftWeightR *  sqrtf((double)tres[2]) / (wG[0]+hG[0]) +
					setting_kfGlobalWeight*setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (wG[0]+hG[0]) +
					setting_kfGlobalWeight*setting_maxAffineWeight * fabs(logf((float)refToFh[0])) ;
// 			if(pic_time_stamp[fh->shell->incoming_id] - pic_time_stamp[coarseTracker->lastRef->shell->incoming_id]>=0.39&&delta>0.3f)
			if(pic_time_stamp[fh->shell->incoming_id] - pic_time_stamp[coarseTracker->lastRef->shell->incoming_id]>=0.45&&delta>0.5f)
				needToMakeKF = true;
// 			if(pic_time_stamp[fh->shell->incoming_id] - pic_time_stamp[coarseTracker->lastRef->shell->incoming_id]>=0.45)
// 				needToMakeKF = true;
// 			LOG(INFO)<<"delta: "<<delta;
// 			LOG(INFO)<<"tres: "<<tres.transpose()<<" refToFh[0]: "<<refToFh[0];

		}
		
// 		LOG(INFO)<<"needToMakeKF: "<<(int)needToMakeKF;
// 		LOG(INFO)<<"coarseTracker->firstCoarseRMSE: "<<coarseTracker->firstCoarseRMSE<<" tres[0]:"<<tres[0];
// 		LOG(INFO)<<"allFrameHistory.size(): "<<allFrameHistory.size();



        for(IOWrap::Output3DWrapper* ow : outputWrapper)
            ow->publishCamPose(fh->shell, &Hcalib);




		lock.unlock();
		deliverTrackedFrame(fh, fh_right, needToMakeKF);
// 		LOG(INFO)<<"fh->worldToCam_evalPT: "<<allFrameHistory[allFrameHistory.size()-1]->camToWorld.translation().transpose();
// 		LOG(INFO)<<"fh->shell->aff_g2l: "<<allFrameHistory[allFrameHistory.size()-1]->aff_g2l.vec().transpose();
// 		LOG(INFO)<<"fh->shell->camToTrackingRef: "<<fh->shell->camToTrackingRef.translation().transpose();
// 		LOG(INFO)<<"fh->shell->trackingRef->camToWorld : "<<fh->shell->trackingRef->camToWorld.translation().transpose();
// 		exit(1);
		
// 		Sophus::Matrix4d T = shell->camToWorld.matrix();
		Sophus::Matrix4d T = T_WD.matrix()*shell->camToWorld.matrix()*T_WD.inverse().matrix();
		savetrajectory_tum(SE3(T),run_time);
// 		savetrajectory(T);
		
		return;
	}
}
void FullSystem::initFirstFrame_imu(FrameHessian* fh){
	int index;
	if(imu_time_stamp.size()>0){
	    for(int i=0;i<imu_time_stamp.size();++i){
		if(imu_time_stamp[i]>=pic_time_stamp[fh->shell->incoming_id]||fabs(imu_time_stamp[i]-pic_time_stamp[fh->shell->incoming_id])<0.001){
		      index = i;
		      break;					  				
		}
	    }
	}
	int index2 = 0;
	if(gt_time_stamp.size()>0){
	    for(int i=0;i<gt_time_stamp.size();++i){
		if(gt_time_stamp[i]>=pic_time_stamp[fh->shell->incoming_id]||fabs(gt_time_stamp[i]-pic_time_stamp[fh->shell->incoming_id])<0.001){
		      index2 = i;
		      break;					  				
		}
	    }
	}
	Vec3 g_b = Vec3::Zero();
	Vec3 g_w;
	g_w<<0,0,-1;
	for(int j=0;j<40;j++){
	    g_b = g_b + m_acc[index-j];
	}
	double norm = g_b.norm();
	g_b = -g_b/norm;
	Vec3 g_c = T_BC.inverse().rotationMatrix()*g_b;

	norm = g_c.norm();
	g_c = g_c/norm;
	Vec3 n = Sophus::SO3::hat(g_c)*g_w;

	norm = n.norm();
	n = n/norm;
	double sin_theta = norm;
	double cos_theta = g_c.dot(g_w);

	Mat33 R_wc = cos_theta*Mat33::Identity()+(1-cos_theta)*n*n.transpose()+sin_theta*Sophus::SO3::hat(n);

	SE3 T_wc(R_wc,Vec3::Zero());
	if(gt_path.size() > 0)
	    T_WR_align = T_wc * T_BC.inverse()*gt_pose[index2].inverse();
	else
	    T_WR_align = SE3();
	
// 	LOG(INFO)<<"first pose: \n"<<T_wc.matrix();
	fh->shell->camToWorld = T_wc;
	fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);

	Mat33 R_wd = Mat33::Identity();

	T_WD = Sim3(RxSO3(1,R_wd),Vec3::Zero());
	T_WD_l = T_WD;
	T_WD_l_half = T_WD;
	state_twd.setZero();
}
void FullSystem::savetrajectory(const Sophus::Matrix4d &T){
	std::ofstream f1;
	std::string dsoposefile = "./data/"+savefile_tail+".txt";
	f1.open(dsoposefile,std::ios::out|std::ios::app);
	f1<<std::fixed<<std::setprecision(9)
	<<T(0,0)<<" "<<T(0,1)<<" "<<T(0,2)<<" "<<T(0,3)<<" "
	<<T(1,0)<<" "<<T(1,1)<<" "<<T(1,2)<<" "<<T(1,3)<<" "
	<<T(2,0)<<" "<<T(2,1)<<" "<<T(2,2)<<" "<<T(2,3)<<std::endl;
	f1.close();
}
void FullSystem::savetrajectory_tum(const SE3 &T, double time){
	
	std::ofstream f1;
	std::string dsoposefile = "./data/"+savefile_tail+".txt";
	f1.open(dsoposefile,std::ios::out|std::ios::app);
	
	Eigen::Quaterniond q = Eigen::Quaterniond(T.rotationMatrix());
	Vec4 q4 = q.coeffs();
	Vec3 t = T.translation();
	
	f1<<std::fixed<<std::setprecision(9)<<time<<" "
	<<t(0)<<" "<<t(1)<<" "<<t(2)<<" "<<q4(0)<<" "<<q4(1)<<" "<<q4(2)<<" "<<q4(3)<<std::endl;
	f1.close();
	
	int index2=0;
	if(gt_path.size()>0 && gt_time_stamp.size()>0){
	    for(int i=0;i<gt_time_stamp.size();++i){
		if(gt_time_stamp[i]>=time||fabs(gt_time_stamp[i]-time)<0.001){
		      index2 = i;
		      break;					  				
		}
	    }
	}
	if(gt_path.size()>0 && fabs(gt_time_stamp[index2]-time)<0.001){
	    std::ofstream f2;
	    std::string gtfile = "./data/"+savefile_tail+"_gt.txt";
	    f2.open(gtfile,std::ios::out|std::ios::app);
	    
	    SE3 gt_C = T_WR_align*gt_pose[index2]*T_BC;
	    q = Eigen::Quaterniond(gt_C.rotationMatrix());
	    q4 = q.coeffs();
	    t = gt_C.translation();
	    f2<<std::fixed<<std::setprecision(9)<<gt_time_stamp[index2]<<" "
	    <<t(0)<<" "<<t(1)<<" "<<t(2)<<" "<<q4(0)<<" "<<q4(1)<<" "<<q4(2)<<" "<<q4(3)<<std::endl;
	    f2.close();
	}
}

void FullSystem::deliverTrackedFrame(FrameHessian* fh, FrameHessian* fh_right, bool needKF)
{


	if(linearizeOperation)
	{
		if(goStepByStep && lastRefStopID != coarseTracker->refFrameID)
		{
			MinimalImageF3 img(wG[0], hG[0], fh->dI);
			IOWrap::displayImage("frameToTrack", &img);
			while(true)
			{
				char k=IOWrap::waitKey(0);
				if(k==' ') break;
				handleKey( k );
			}
			lastRefStopID = coarseTracker->refFrameID;
		}
		else handleKey( IOWrap::waitKey(1) );



		if(needKF) makeKeyFrame(fh,fh_right);
		else makeNonKeyFrame(fh,fh_right);
	}
	else
	{
		boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
		unmappedTrackedFrames.push_back(fh);
		if(needKF) needNewKFAfter=fh->shell->trackingRef->id;
		trackedFrameSignal.notify_all();

		while(coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1 )
		{
			mappedFrameSignal.wait(lock);
		}

		lock.unlock();
	}
}

void FullSystem::mappingLoop()
{
	boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);

	while(runMapping)
	{
		while(unmappedTrackedFrames.size()==0)
		{
			trackedFrameSignal.wait(lock);
			if(!runMapping) return;
		}

		FrameHessian* fh = unmappedTrackedFrames.front();
		unmappedTrackedFrames.pop_front();
		FrameHessian* fh_right = unmappedTrackedFrames_right.front();
		unmappedTrackedFrames_right.pop_front();


		// guaranteed to make a KF for the very first two tracked frames.
		if(allKeyFramesHistory.size() <= 2)
		{
			lock.unlock();
			makeKeyFrame(fh,fh_right);
			lock.lock();
			mappedFrameSignal.notify_all();
			continue;
		}

		if(unmappedTrackedFrames.size() > 3)
			needToKetchupMapping=true;


		if(unmappedTrackedFrames.size() > 0) // if there are other frames to tracke, do that first.
		{
			lock.unlock();
			makeNonKeyFrame(fh,fh_right);
			lock.lock();

			if(needToKetchupMapping && unmappedTrackedFrames.size() > 0)
			{
				FrameHessian* fh = unmappedTrackedFrames.front();
				unmappedTrackedFrames.pop_front();
				{
					boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
					assert(fh->shell->trackingRef != 0);
					fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
					fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
				}
				delete fh;
			}

		}
		else
		{
			if(setting_realTimeMaxKF || needNewKFAfter >= frameHessians.back()->shell->id)
			{
				lock.unlock();
				makeKeyFrame(fh,fh_right);
				needToKetchupMapping=false;
				lock.lock();
			}
			else
			{
				lock.unlock();
				makeNonKeyFrame(fh,fh_right);
				lock.lock();
			}
		}
		mappedFrameSignal.notify_all();
	}
	printf("MAPPING FINISHED!\n");
}

void FullSystem::blockUntilMappingIsFinished()
{
	boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
	runMapping = false;
	trackedFrameSignal.notify_all();
	lock.unlock();

	mappingThread.join();

}

void FullSystem::makeNonKeyFrame( FrameHessian* fh, FrameHessian* fh_right)
{
	// needs to be set by mapping thread. no lock required since we are in mapping thread.
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		assert(fh->shell->trackingRef != 0);
		fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
		fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
	}
	
	traceNewCoarse(fh);

// 	traceNewCoarseNonKey(fh, fh_right);
	delete fh;
	delete fh_right;
}

void FullSystem::makeKeyFrame( FrameHessian* fh, FrameHessian* fh_right)
{
	// needs to be set by mapping thread
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		assert(fh->shell->trackingRef != 0);
		fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
		fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
	}
// 	LOG(INFO)<<"make keyframe";
	traceNewCoarse(fh);
// 	traceNewCoarseKey(fh, fh_right);
// 	traceNewCoarseNonKey(fh, fh_right);

	boost::unique_lock<boost::mutex> lock(mapMutex);

	// =========================== Flag Frames to be Marginalized. =========================
	flagFramesForMarginalization(fh);


	// =========================== add New Frame to Hessian Struct. =========================
	fh->idx = frameHessians.size();
// 	fh_right->idx = frameHessians_right.size();
	frameHessians.push_back(fh);
// 	frameHessians_right.push_back(fh_right);
	fh->frameID = allKeyFramesHistory.size();
	fh->frame_right->frameID = 10000+allKeyFramesHistory.size();
// 	fh->frame_right->frameID = allKeyFramesHistory.size();
	
	allKeyFramesHistory.push_back(fh->shell);
	ef->insertFrame(fh, &Hcalib);

	setPrecalcValues();



	// =========================== add new residuals for old points =========================
	int numFwdResAdde=0;
	for(FrameHessian* fh1 : frameHessians)		// go through all active frames
	{
		if(fh1 == fh) continue;
		for(PointHessian* ph : fh1->pointHessians)
		{
			PointFrameResidual* r = new PointFrameResidual(ph, fh1, fh);
			r->setState(ResState::IN);
			ph->residuals.push_back(r);
			ef->insertResidual(r);
			ph->lastResiduals[1] = ph->lastResiduals[0];
			ph->lastResiduals[0] = std::pair<PointFrameResidual*, ResState>(r, ResState::IN);
			numFwdResAdde+=1;
		}
	}



	// =========================== Activate Points (& flag for marginalization). =========================
	activatePointsMT();
	ef->makeIDX();



	// =========================== OPTIMIZE ALL =========================

	fh->frameEnergyTH = frameHessians.back()->frameEnergyTH;
// 	LOG(INFO)<<"optimize start";
	float rmse = optimize(setting_maxOptIterations);
// 	LOG(INFO)<<"rmse: "<<rmse;




	// =========================== Figure Out if INITIALIZATION FAILED =========================
	if(allKeyFramesHistory.size() <= 4)
	{
// 		LOG(INFO)<<"allKeyFramesHistory.size(): "<<allKeyFramesHistory.size()<<" rmse: "<<rmse;
		if(allKeyFramesHistory.size()==2 && rmse > 20*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
		if(allKeyFramesHistory.size()==3 && rmse > 13*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
		if(allKeyFramesHistory.size()==4 && rmse > 9*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
	}



    if(isLost) return;



// 	LOG(INFO)<<"remove outliers.";
	// =========================== REMOVE OUTLIER =========================
	removeOutliers();




	{
		boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
		coarseTracker_forNewKF->makeK(&Hcalib);
		coarseTracker_forNewKF->setCoarseTrackingRef(frameHessians, fh_right, Hcalib);



        coarseTracker_forNewKF->debugPlotIDepthMap(&minIdJetVisTracker, &maxIdJetVisTracker, outputWrapper);
        coarseTracker_forNewKF->debugPlotIDepthMapFloat(outputWrapper);
	}


	debugPlot("post Optimize");





// 	LOG(INFO)<<"flagPointsForRemoval";
	// =========================== (Activate-)Marginalize Points =========================
	flagPointsForRemoval();
// 	LOG(INFO)<<"ef->dropPointsF()";
	ef->dropPointsF();
// 	LOG(INFO)<<"after dropPoints: "<<ef->nPoints;
// 	LOG(INFO)<<"getNullspaces";
	getNullspaces(
			ef->lastNullspaces_pose,
			ef->lastNullspaces_scale,
			ef->lastNullspaces_affA,
			ef->lastNullspaces_affB);
// 	LOG(INFO)<<"ef->marginalizePointsF();";
	ef->marginalizePointsF();


// 	LOG(INFO)<<"makeNewTraces";
	// =========================== add new Immature points & new residuals =========================
	makeNewTraces(fh, 0);

// 	LOG(INFO)<<"makeNewTraces end";



    for(IOWrap::Output3DWrapper* ow : outputWrapper)
    {
	ow->publishGraph(ef->connectivityMap);
        ow->publishKeyframes(frameHessians, false, &Hcalib);
    }


// 	LOG(INFO)<<"marginalizeFrame";
	// =========================== Marginalize Frames =========================

	for(unsigned int i=0;i<frameHessians.size();i++)
		if(frameHessians[i]->flaggedForMarginalization)
			{marginalizeFrame(frameHessians[i]); i=0;}
// 	LOG(INFO)<<"make key end";
// 	delete fh_right;

// 	printLogLine();
    //printEigenValLine();
	

}


void FullSystem::initializeFromInitializer(FrameHessian* newFrame)
{
	boost::unique_lock<boost::mutex> lock(mapMutex);

	// add firstframe.
	FrameHessian* firstFrame = coarseInitializer->firstFrame;
	firstFrame->idx = frameHessians.size();
	frameHessians.push_back(firstFrame);
	firstFrame->frameID = allKeyFramesHistory.size();
	firstFrame->frame_right->frameID = 10000+allKeyFramesHistory.size();
// 	firstFrame->frame_right->frameID = allKeyFramesHistory.size();
	allKeyFramesHistory.push_back(firstFrame->shell);
	ef->insertFrame(firstFrame, &Hcalib);
	setPrecalcValues();

	FrameHessian* firstFrameRight = coarseInitializer->firstFrame_right;

	//int numPointsTotal = makePixelStatus(firstFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
	//int numPointsTotal = pixelSelector->makeMaps(firstFrame->dIp, selectionMap,setting_desiredDensity);

	firstFrame->pointHessians.reserve(wG[0]*hG[0]*0.2f);
	firstFrame->pointHessiansMarginalized.reserve(wG[0]*hG[0]*0.2f);
	firstFrame->pointHessiansOut.reserve(wG[0]*hG[0]*0.2f);

	float idepthStereo = 0;
	float sumID=1e-5, numID=1e-5;
	for(int i=0;i<coarseInitializer->numPoints[0];i++)
	{
		sumID += coarseInitializer->points[0][i].iR;
		numID++;
	}
	float rescaleFactor = 1 / (sumID / numID);

	// randomly sub-select the points I need.
	float keepPercentage = setting_desiredPointDensity / coarseInitializer->numPoints[0];

    if(!setting_debugout_runquiet)
        printf("Initialization: keep %.1f%% (need %d, have %d)!\n", 100*keepPercentage,
                (int)(setting_desiredPointDensity), coarseInitializer->numPoints[0] );
	if(use_stereo){
	    for(int i=0;i<coarseInitializer->numPoints[0];i++)
	    {
		    if(rand()/(float)RAND_MAX > keepPercentage) continue;

		    Pnt* point = coarseInitializer->points[0]+i;
		    ImmaturePoint* pt = new ImmaturePoint(point->u+0.5f,point->v+0.5f,firstFrame,point->my_type, &Hcalib);
		    
		    pt->u_stereo = pt->u;
		    pt->v_stereo = pt->v;
		    pt->idepth_min_stereo = 0;
		    pt->idepth_max_stereo = NAN;

		    pt->traceStereo(firstFrameRight, &Hcalib, 1);

		    pt->idepth_min = pt->idepth_min_stereo;
		    pt->idepth_max = pt->idepth_max_stereo;
		    idepthStereo = pt->idepth_stereo;
		    
		    double d_mid = 0.5*(1/pt->idepth_min+1/pt->idepth_max);
		    if(!std::isfinite(pt->energyTH) || !std::isfinite(pt->idepth_min) || !std::isfinite(pt->idepth_max)
				    /*||d_mid<0||d_mid>100*/|| pt->idepth_min < 0 || pt->idepth_max < 0)
		    {
			delete pt;
			continue;

		    }
		    PointHessian* ph = new PointHessian(pt, &Hcalib);
		    delete pt;
		    if(!std::isfinite(ph->energyTH)) {delete ph; continue;}
		    
		    ph->setIdepthScaled(idepthStereo);
		    ph->setIdepthZero(idepthStereo);
		    ph->hasDepthPrior=true;
		    ph->setPointStatus(PointHessian::ACTIVE);

    // 		ph->setIdepthScaled(point->iR*rescaleFactor);
    // 		ph->setIdepthZero(ph->idepth);
    // 		ph->hasDepthPrior=true;
    // 		ph->setPointStatus(PointHessian::ACTIVE);

		    firstFrame->pointHessians.push_back(ph);
		    ef->insertPoint(ph);
		    PointFrameResidual* r = new PointFrameResidual(ph, ph->host, ph->host->frame_right);
		    r->state_NewEnergy = r->state_energy = 0;
		    r->state_NewState = ResState::OUTLIER;
		    r->setState(ResState::IN);
		    r->stereoResidualFlag = true;
		    ph->residuals.push_back(r);
		    ef->insertResidual(r);
	      
	      
	      
    // 		if(rand()/(float)RAND_MAX > keepPercentage) continue;
    // 
    // 		Pnt* point = coarseInitializer->points[0]+i;
    // 		ImmaturePoint* pt = new ImmaturePoint(point->u+0.5f,point->v+0.5f,firstFrame,point->my_type, &Hcalib);
    // 
    // 		if(!std::isfinite(pt->energyTH)) { delete pt; continue; }
    // 
    // 
    // 		pt->idepth_max=pt->idepth_min=1;
    // 		PointHessian* ph = new PointHessian(pt, &Hcalib);
    // 		delete pt;
    // 		if(!std::isfinite(ph->energyTH)) {delete ph; continue;}
    // 
    // 		ph->setIdepthScaled(point->iR*rescaleFactor);
    // 		ph->setIdepthZero(ph->idepth);
    // 		ph->hasDepthPrior=true;
    // 		ph->setPointStatus(PointHessian::ACTIVE);
    // 
    // 		firstFrame->pointHessians.push_back(ph);
    // 		ef->insertPoint(ph);
	    }
	}
	else{
	    for(int i=0;i<coarseInitializer->numPoints[0];i++){
		if(rand()/(float)RAND_MAX > keepPercentage) continue;
    
    		Pnt* point = coarseInitializer->points[0]+i;
    		ImmaturePoint* pt = new ImmaturePoint(point->u+0.5f,point->v+0.5f,firstFrame,point->my_type, &Hcalib);
    
    		if(!std::isfinite(pt->energyTH)) { delete pt; continue; }
    
    
    		pt->idepth_max=pt->idepth_min=1;
    		PointHessian* ph = new PointHessian(pt, &Hcalib);
    		delete pt;
    		if(!std::isfinite(ph->energyTH)) {delete ph; continue;}
    
    		ph->setIdepthScaled(point->iR*rescaleFactor);
    		ph->setIdepthZero(ph->idepth);
    		ph->hasDepthPrior=true;
    		ph->setPointStatus(PointHessian::ACTIVE);
    
    		firstFrame->pointHessians.push_back(ph);
    		ef->insertPoint(ph);
	    }
	}

	SE3 firstToNew = coarseInitializer->thisToNext;
	firstToNew.translation() /= rescaleFactor;


	// really no lock required, as we are initializing.
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
// 		firstFrame->shell->camToWorld = SE3();
// 		firstFrame->shell->aff_g2l = AffLight(0,0);
// 		firstFrame->setEvalPT_scaled(firstFrame->shell->camToWorld.inverse(),firstFrame->shell->aff_g2l);
		firstFrame->shell->trackingRef=0;
		firstFrame->shell->camToTrackingRef = SE3();
// 		LOG(INFO)<<"firstFrame pose: \n"<<firstFrame->shell->camToWorld.matrix();

		newFrame->shell->camToWorld = firstFrame->shell->camToWorld*firstToNew.inverse();
		newFrame->shell->aff_g2l = AffLight(0,0);
		newFrame->setEvalPT_scaled(newFrame->shell->camToWorld.inverse(),newFrame->shell->aff_g2l);
		newFrame->shell->trackingRef = firstFrame->shell;
		newFrame->shell->camToTrackingRef = firstToNew.inverse();
		
// 		LOG(INFO)<<"newFrame pose: \n"<<newFrame->shell->camToWorld.matrix();

	}

	initialized=true;
	printf("INITIALIZE FROM INITIALIZER (%d pts)!\n", (int)firstFrame->pointHessians.size());
// 	if((int)firstFrame->pointHessians.size()<600)initFailed=true;
}

void FullSystem::makeNewTraces(FrameHessian* newFrame, float* gtDepth)
{
	pixelSelector->allowFast = true;
	//int numPointsTotal = makePixelStatus(newFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
	int numPointsTotal = pixelSelector->makeMaps(newFrame, selectionMap,setting_desiredImmatureDensity);

	newFrame->pointHessians.reserve(numPointsTotal*1.2f);
	//fh->pointHessiansInactive.reserve(numPointsTotal*1.2f);
	newFrame->pointHessiansMarginalized.reserve(numPointsTotal*1.2f);
	newFrame->pointHessiansOut.reserve(numPointsTotal*1.2f);


	for(int y=patternPadding+1;y<hG[0]-patternPadding-2;y++)
	for(int x=patternPadding+1;x<wG[0]-patternPadding-2;x++)
	{
		int i = x+y*wG[0];
		if(selectionMap[i]==0) continue;

		ImmaturePoint* impt = new ImmaturePoint(x,y,newFrame, selectionMap[i], &Hcalib);
		if(!std::isfinite(impt->energyTH)) delete impt;
		else newFrame->immaturePoints.push_back(impt);

	}
	//printf("MADE %d IMMATURE POINTS!\n", (int)newFrame->immaturePoints.size());

}



void FullSystem::setPrecalcValues()
{
	for(FrameHessian* fh : frameHessians)
	{
		fh->targetPrecalc.resize(frameHessians.size());
		for(unsigned int i=0;i<frameHessians.size();i++)
			fh->targetPrecalc[i].set(fh, frameHessians[i], &Hcalib);
	}

	ef->setDeltaF(&Hcalib);
}


void FullSystem::printLogLine()
{
	if(frameHessians.size()==0) return;

    if(!setting_debugout_runquiet)
        printf("LOG %d: %.3f fine. Res: %d A, %d L, %d M; (%'d / %'d) forceDrop. a=%f, b=%f. Window %d (%d)\n",
                allKeyFramesHistory.back()->id,
                statistics_lastFineTrackRMSE,
                ef->resInA,
                ef->resInL,
                ef->resInM,
                (int)statistics_numForceDroppedResFwd,
                (int)statistics_numForceDroppedResBwd,
                allKeyFramesHistory.back()->aff_g2l.a,
                allKeyFramesHistory.back()->aff_g2l.b,
                frameHessians.back()->shell->id - frameHessians.front()->shell->id,
                (int)frameHessians.size());


	if(!setting_logStuff) return;

	if(numsLog != 0)
	{
		(*numsLog) << allKeyFramesHistory.back()->id << " "  <<
				statistics_lastFineTrackRMSE << " "  <<
				(int)statistics_numCreatedPoints << " "  <<
				(int)statistics_numActivatedPoints << " "  <<
				(int)statistics_numDroppedPoints << " "  <<
				(int)statistics_lastNumOptIts << " "  <<
				ef->resInA << " "  <<
				ef->resInL << " "  <<
				ef->resInM << " "  <<
				statistics_numMargResFwd << " "  <<
				statistics_numMargResBwd << " "  <<
				statistics_numForceDroppedResFwd << " "  <<
				statistics_numForceDroppedResBwd << " "  <<
				frameHessians.back()->aff_g2l().a << " "  <<
				frameHessians.back()->aff_g2l().b << " "  <<
				frameHessians.back()->shell->id - frameHessians.front()->shell->id << " "  <<
				(int)frameHessians.size() << " "  << "\n";
		numsLog->flush();
	}


}



void FullSystem::printEigenValLine()
{
	if(!setting_logStuff) return;
	if(ef->lastHS.rows() < 12) return;


	MatXX Hp = ef->lastHS.bottomRightCorner(ef->lastHS.cols()-CPARS,ef->lastHS.cols()-CPARS);
	MatXX Ha = ef->lastHS.bottomRightCorner(ef->lastHS.cols()-CPARS,ef->lastHS.cols()-CPARS);
	int n = Hp.cols()/8;
	assert(Hp.cols()%8==0);

	// sub-select
	for(int i=0;i<n;i++)
	{
		MatXX tmp6 = Hp.block(i*8,0,6,n*8);
		Hp.block(i*6,0,6,n*8) = tmp6;

		MatXX tmp2 = Ha.block(i*8+6,0,2,n*8);
		Ha.block(i*2,0,2,n*8) = tmp2;
	}
	for(int i=0;i<n;i++)
	{
		MatXX tmp6 = Hp.block(0,i*8,n*8,6);
		Hp.block(0,i*6,n*8,6) = tmp6;

		MatXX tmp2 = Ha.block(0,i*8+6,n*8,2);
		Ha.block(0,i*2,n*8,2) = tmp2;
	}

	VecX eigenvaluesAll = ef->lastHS.eigenvalues().real();
	VecX eigenP = Hp.topLeftCorner(n*6,n*6).eigenvalues().real();
	VecX eigenA = Ha.topLeftCorner(n*2,n*2).eigenvalues().real();
	VecX diagonal = ef->lastHS.diagonal();

	std::sort(eigenvaluesAll.data(), eigenvaluesAll.data()+eigenvaluesAll.size());
	std::sort(eigenP.data(), eigenP.data()+eigenP.size());
	std::sort(eigenA.data(), eigenA.data()+eigenA.size());

	int nz = std::max(100,setting_maxFrames*10);

	if(eigenAllLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenvaluesAll.size()) = eigenvaluesAll;
		(*eigenAllLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenAllLog->flush();
	}
	if(eigenALog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenA.size()) = eigenA;
		(*eigenALog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenALog->flush();
	}
	if(eigenPLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenP.size()) = eigenP;
		(*eigenPLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenPLog->flush();
	}

	if(DiagonalLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = diagonal;
		(*DiagonalLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		DiagonalLog->flush();
	}

	if(variancesLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = ef->lastHS.inverse().diagonal();
		(*variancesLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		variancesLog->flush();
	}

	std::vector<VecX> &nsp = ef->lastNullspaces_forLogging;
	(*nullspacesLog) << allKeyFramesHistory.back()->id << " ";
	for(unsigned int i=0;i<nsp.size();i++)
		(*nullspacesLog) << nsp[i].dot(ef->lastHS * nsp[i]) << " " << nsp[i].dot(ef->lastbS) << " " ;
	(*nullspacesLog) << "\n";
	nullspacesLog->flush();

}

void FullSystem::printFrameLifetimes()
{
	if(!setting_logStuff) return;


	boost::unique_lock<boost::mutex> lock(trackMutex);

	std::ofstream* lg = new std::ofstream();
	lg->open("logs/lifetimeLog.txt", std::ios::trunc | std::ios::out);
	lg->precision(15);

	for(FrameShell* s : allFrameHistory)
	{
		(*lg) << s->id
			<< " " << s->marginalizedAt
			<< " " << s->statistics_goodResOnThis
			<< " " << s->statistics_outlierResOnThis
			<< " " << s->movedByOpt;



		(*lg) << "\n";
	}





	lg->close();
	delete lg;

}


void FullSystem::printEvalLine()
{
	return;
}





}
