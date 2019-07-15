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

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/ImmaturePoint.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "IOWrapper/ImageRW.h"
#include <algorithm>

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{


template<int b, typename T>
T* allocAligned(int size, std::vector<T*> &rawPtrVec)
{
    const int padT = 1 + ((1 << b)/sizeof(T));
    T* ptr = new T[size + padT];
    rawPtrVec.push_back(ptr);
    T* alignedPtr = (T*)(( ((uintptr_t)(ptr+padT)) >> b) << b);
    return alignedPtr;
}

double CoarseTracker::calcIMUResAndGS(Mat66 &H_out, Vec6 &b_out, SE3 &refToNew, const IMUPreintegrator &IMU_preintegrator, Vec9 &res_PVPhi, double PointEnergy, double imu_track_weight){
    

    
    Mat44 M_DCi = lastRef->shell->camToWorld.matrix();
    Mat44 M_WD = T_WD.matrix();
    Mat44 M_WB = M_WD*M_DCi*M_WD.inverse()*T_BC.inverse().matrix();
    SE3 T_WB(M_WB);
    Mat33 R_WB = T_WB.rotationMatrix();
    Vec3 t_WB = T_WB.translation();
    
    SE3 newToRef = refToNew.inverse();    
    Mat44 M_DCj = (lastRef->shell->camToWorld * newToRef).matrix();
    Mat44 M_WBj = M_WD*M_DCj*M_WD.inverse()*T_BC.inverse().matrix();
    SE3 T_WBj(M_WBj);
    Mat33 R_WBj = T_WBj.rotationMatrix();
    Vec3 t_WBj = T_WBj.translation();
    
    double dt = IMU_preintegrator.getDeltaTime();
//     LOG(INFO)<<"dt: "<<dt;
    H_out = Mat66::Zero();
    b_out = Vec6::Zero();
    if(dt>0.5){
	return 0;
    }
    Vec3 g_w;
    g_w << 0,0,-1;
    g_w = g_w*G_norm;
    
  

    Vec3 so3 = IMU_preintegrator.getJRBiasg()*lastRef->delta_bias_g;
    double theta = so3.norm();
    Mat33 R_temp = Mat33::Identity(); 
    R_temp = SO3::exp(IMU_preintegrator.getJRBiasg()*lastRef->delta_bias_g).matrix();
    
    Mat33 res_R = (IMU_preintegrator.getDeltaR()*R_temp).transpose()*R_WB.transpose()*R_WBj;
//     LOG(INFO)<<"res_R: \n"<<res_R;
    
    Vec3 res_phi = SO3(res_R).log();
    
//     LOG(INFO)<<"res_phi: "<<res_phi.transpose();
//     
//     LOG(INFO)<<"IMU_preintegrator.getDeltaV(): "<<IMU_preintegrator.getDeltaV().transpose();
	newFrame->velocity = R_WB*(R_WB.transpose()*(lastRef->velocity+g_w*dt)+
		 (IMU_preintegrator.getDeltaV()+IMU_preintegrator.getJVBiasa()*lastRef->delta_bias_a+IMU_preintegrator.getJVBiasg()*lastRef->delta_bias_g));
// 	Vec3 res_v = R_WB.transpose()*(newFrame->velocity-lastRef->velocity-g_w*dt)-
// 		 (IMU_preintegrator.getDeltaV()+IMU_preintegrator.getJVBiasa()*lastRef->delta_bias_a+IMU_preintegrator.getJVBiasg()*lastRef->delta_bias_g);
// 	LOG(INFO)<<"res_v: "<<res_v.transpose();
// 	LOG(INFO)<<std::fixed<<std::setprecision(14)<<"newFrame time: "<<pic_time_stamp[newFrame->shell->incoming_id];
// 	int index2;
// 	if(gt_time_stamp.size()>0){
// 	    for(int i=0;i<gt_time_stamp.size();++i){
// 		if(gt_time_stamp[i]>=pic_time_stamp[newFrame->shell->incoming_id]||fabs(gt_time_stamp[i]-pic_time_stamp[newFrame->shell->incoming_id])<0.001){
// 		      index2 = i;
// 		      break;					  				
// 		}
// 	    }
// 	}
// 	newFrame->velocity = gt_velocity[index2];
// 	LOG(INFO)<<"newFrame->velocity: "<<newFrame->velocity.transpose();
	newFrame->shell->velocity = newFrame->velocity;
//     LOG(INFO)<<"lastRef->velocity: "<<lastRef->velocity.transpose();
//     LOG(INFO)<<"newFrame->velocity: "<<newFrame->velocity.transpose();
//     LOG(INFO)<<"(R_WB.transpose()*g_w).transpose()"<<(R_WB.transpose()*g_w).transpose();
	
    Vec3 res_p = R_WB.transpose()*(t_WBj-t_WB-lastRef->velocity*dt-0.5*g_w*dt*dt)-
		 (IMU_preintegrator.getDeltaP()+IMU_preintegrator.getJPBiasa()*lastRef->delta_bias_a+IMU_preintegrator.getJPBiasg()*lastRef->delta_bias_g);
	
	
//     LOG(INFO)<<"res_p: "<<res_p.transpose();
//     exit(1);
// 
//     LOG(INFO)<<"newFrame->velocity: "<<newFrame->velocity.transpose();
    Mat99 Cov = IMU_preintegrator.getCovPVPhi();
//     LOG(INFO)<<"Cov: \n"<<Cov;
    
//     Vec9 res_PVPhi;
    res_PVPhi.block(0,0,3,1) = res_p;
//     res_PVPhi.block(0,0,3,1) = Vec3::Zero();
    res_PVPhi.block(3,0,3,1) = Vec3::Zero();
    res_PVPhi.block(6,0,3,1) = res_phi;
    
//     double lambda = 0.03;
    double res = imu_track_weight*imu_track_weight*res_PVPhi.transpose() * Cov.inverse() * res_PVPhi;
//     LOG(INFO)<<"res: "<<res<<" PointEnergy: "<<PointEnergy;
//     LOG(INFO)<<"Cov.inverse(): \n"<<Cov.inverse();
//     double bei = sqrt(PointEnergy/res);
//     bei /= imu_lambda;
    
    Mat33 J_resPhi_phi_j = IMU_preintegrator.JacobianRInv(res_phi);
    Mat33 J_resV_v_j = R_WB.transpose();
    Mat33 J_resP_p_j = R_WB.transpose()*R_WBj;

    
    Mat66 J_imu1 = Mat66::Zero();
    J_imu1.block(0,0,3,3) = J_resP_p_j;
    J_imu1.block(3,3,3,3) = J_resPhi_phi_j;
//     J_imu1.block(6,6,3,3) = J_resV_v_j;
    Mat66 Weight = Mat66::Zero();
    Weight.block(0,0,3,3) = Cov.block(0,0,3,3);
    Weight.block(3,3,3,3) = Cov.block(6,6,3,3);
//     Weight.block(6,6,3,3) = Cov.block(3,3,3,3);
    Mat66 Weight2 = Mat66::Zero();
    for(int i=0;i<6;++i){
	Weight2(i,i) = Weight(i,i);
    }
    Weight = Weight2.inverse();
    Weight *=(imu_track_weight*imu_track_weight);
//     for(int i=0;i<3;++i){
//        Weight(i,i) *= 0.2;
//     }
//     Weight *=(bei*bei);
    
    Vec6 b_1 = Vec6::Zero();
    b_1.block(0,0,3,1) = res_p;
    b_1.block(3,0,3,1) = res_phi;
//     b_1.block(6,0,3,1) = res_v;
    
    Mat44 T_temp = T_BC.matrix()*T_WD.matrix()*M_DCj.inverse();
    Mat66 J_rel = (-1*Sim3(T_temp).Adj()).block(0,0,6,6);
//     Mat44 T_temp = T_BC.matrix()*M_DCj.inverse();
//     Mat66 J_rel = (-1*SE3(T_temp).Adj());
    Mat66 J_xi_tw_th = SE3(M_DCi).Adj();
//     LOG(INFO)<<"-1*Sim3(T_temp).Adj: "<<-1*Sim3(T_temp).Adj().matrix();
    
//     LOG(INFO)<<"J_imu1: \n"<<J_imu1;
    
    
//     Mat99 J_rel = Mat99::Identity();
//     J_rel.block(0,0,6,6) = M_rel;
    Mat66 J_xi_r_l = refToNew.Adj().inverse();
    Mat66 J_2 = Mat66::Zero();
    J_2 = J_imu1*J_rel*J_xi_tw_th*J_xi_r_l;

    H_out = J_2.transpose()*Weight*J_2;
    b_out = J_2.transpose()*Weight*b_1;
    
    H_out.block<6,3>(0,0) *= SCALE_XI_TRANS;
    H_out.block<6,3>(0,3) *= SCALE_XI_ROT;
    H_out.block<3,6>(0,0) *= SCALE_XI_TRANS;
    H_out.block<3,6>(3,0) *= SCALE_XI_ROT;
//     H_out.block<9,3>(0,6) *= SCALE_V;
//     H_out.block<3,9>(6,0) *= SCALE_V;
    
    b_out.segment<3>(0) *= SCALE_XI_TRANS;
    b_out.segment<3>(3) *= SCALE_XI_ROT;
//     b_out.segment<3>(6) *= SCALE_V;
//     LOG(INFO)<<"J_2: \n"<<J_2;
//     LOG(INFO)<<"weight: \n"<<Weight;
//     LOG(INFO)<<"b_1: "<<b_1.transpose();
//     LOG(INFO)<<"H_out: \n"<<H_out;
//     LOG(INFO)<<"b_out: \n"<<b_out.transpose();
//     exit(1);
    return res;
}
CoarseTracker::CoarseTracker(int ww, int hh) : lastRef_aff_g2l(0,0)
{
	// make coarse tracking templates.
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		int wl = ww>>lvl;
        int hl = hh>>lvl;

        idepth[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
        weightSums[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
        weightSums_bak[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);

        pc_u[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
        pc_v[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
        pc_idepth[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
        pc_color[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);

	}

	// warped buffers
    buf_warped_idepth = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_u = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_v = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_dx = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_dy = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_residual = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_weight = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_refColor = allocAligned<4,float>(ww*hh, ptrToDelete);


	newFrame = 0;
	lastRef = 0;
	debugPlot = debugPrint = true;
	w[0]=h[0]=0;
	refFrameID=-1;
}
CoarseTracker::~CoarseTracker()
{
    for(float* ptr : ptrToDelete)
        delete[] ptr;
    ptrToDelete.clear();
}

void CoarseTracker::makeK(CalibHessian* HCalib)
{
	w[0] = wG[0];
	h[0] = hG[0];

	fx[0] = HCalib->fxl();
	fy[0] = HCalib->fyl();
	cx[0] = HCalib->cxl();
	cy[0] = HCalib->cyl();

	for (int level = 1; level < pyrLevelsUsed; ++ level)
	{
		w[level] = w[0] >> level;
		h[level] = h[0] >> level;
		fx[level] = fx[level-1] * 0.5;
		fy[level] = fy[level-1] * 0.5;
		cx[level] = (cx[0] + 0.5) / ((int)1<<level) - 0.5;
		cy[level] = (cy[0] + 0.5) / ((int)1<<level) - 0.5;
	}

	for (int level = 0; level < pyrLevelsUsed; ++ level)
	{
		K[level]  << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
		Ki[level] = K[level].inverse();
		fxi[level] = Ki[level](0,0);
		fyi[level] = Ki[level](1,1);
		cxi[level] = Ki[level](0,2);
		cyi[level] = Ki[level](1,2);
	}
}

void CoarseTracker::makeCoarseDepthForFirstFrame(FrameHessian* fh)
{
    // make coarse tracking templates for latstRef.
    memset(idepth[0], 0, sizeof(float)*w[0]*h[0]);
    memset(weightSums[0], 0, sizeof(float)*w[0]*h[0]);

    for(PointHessian* ph : fh->pointHessians)
    {
        int u = ph->u + 0.5f;
        int v = ph->v + 0.5f;
        float new_idepth = ph->idepth;
        float weight = sqrtf(1e-3 / (ph->efPoint->HdiF+1e-12));

        idepth[0][u+w[0]*v] += new_idepth *weight;
        weightSums[0][u+w[0]*v] += weight;

    }

    for(int lvl=1; lvl<pyrLevelsUsed; lvl++)
    {
        int lvlm1 = lvl-1;
        int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

        float* idepth_l = idepth[lvl];
        float* weightSums_l = weightSums[lvl];

        float* idepth_lm = idepth[lvlm1];
        float* weightSums_lm = weightSums[lvlm1];

        for(int y=0;y<hl;y++)
            for(int x=0;x<wl;x++)
            {
                int bidx = 2*x   + 2*y*wlm1;
                idepth_l[x + y*wl] = 		idepth_lm[bidx] +
                                            idepth_lm[bidx+1] +
                                            idepth_lm[bidx+wlm1] +
                                            idepth_lm[bidx+wlm1+1];

                weightSums_l[x + y*wl] = 	weightSums_lm[bidx] +
                                            weightSums_lm[bidx+1] +
                                            weightSums_lm[bidx+wlm1] +
                                            weightSums_lm[bidx+wlm1+1];
            }
    }

    // dilate idepth by 1.
    for(int lvl=0; lvl<2; lvl++)
    {
        int numIts = 1;


        for(int it=0;it<numIts;it++)
        {
            int wh = w[lvl]*h[lvl]-w[lvl];
            int wl = w[lvl];
            float* weightSumsl = weightSums[lvl];
            float* weightSumsl_bak = weightSums_bak[lvl];
            memcpy(weightSumsl_bak, weightSumsl, w[lvl]*h[lvl]*sizeof(float));
            float* idepthl = idepth[lvl];	// dont need to make a temp copy of depth, since I only
            // read values with weightSumsl>0, and write ones with weightSumsl<=0.
            for(int i=w[lvl];i<wh;i++)
            {
                if(weightSumsl_bak[i] <= 0)
                {
                    float sum=0, num=0, numn=0;
                    if(weightSumsl_bak[i+1+wl] > 0) { sum += idepthl[i+1+wl]; num+=weightSumsl_bak[i+1+wl]; numn++;}
                    if(weightSumsl_bak[i-1-wl] > 0) { sum += idepthl[i-1-wl]; num+=weightSumsl_bak[i-1-wl]; numn++;}
                    if(weightSumsl_bak[i+wl-1] > 0) { sum += idepthl[i+wl-1]; num+=weightSumsl_bak[i+wl-1]; numn++;}
                    if(weightSumsl_bak[i-wl+1] > 0) { sum += idepthl[i-wl+1]; num+=weightSumsl_bak[i-wl+1]; numn++;}
                    if(numn>0) {idepthl[i] = sum/numn; weightSumsl[i] = num/numn;}
                }
            }
        }
    }


    // dilate idepth by 1 (2 on lower levels).
    for(int lvl=2; lvl<pyrLevelsUsed; lvl++)
    {
        int wh = w[lvl]*h[lvl]-w[lvl];
        int wl = w[lvl];
        float* weightSumsl = weightSums[lvl];
        float* weightSumsl_bak = weightSums_bak[lvl];
        memcpy(weightSumsl_bak, weightSumsl, w[lvl]*h[lvl]*sizeof(float));
        float* idepthl = idepth[lvl];	// dotnt need to make a temp copy of depth, since I only
        // read values with weightSumsl>0, and write ones with weightSumsl<=0.
        for(int i=w[lvl];i<wh;i++)
        {
            if(weightSumsl_bak[i] <= 0)
            {
                float sum=0, num=0, numn=0;
                if(weightSumsl_bak[i+1] > 0) { sum += idepthl[i+1]; num+=weightSumsl_bak[i+1]; numn++;}
                if(weightSumsl_bak[i-1] > 0) { sum += idepthl[i-1]; num+=weightSumsl_bak[i-1]; numn++;}
                if(weightSumsl_bak[i+wl] > 0) { sum += idepthl[i+wl]; num+=weightSumsl_bak[i+wl]; numn++;}
                if(weightSumsl_bak[i-wl] > 0) { sum += idepthl[i-wl]; num+=weightSumsl_bak[i-wl]; numn++;}
                if(numn>0) {idepthl[i] = sum/numn; weightSumsl[i] = num/numn;}
            }
        }
    }


    // normalize idepths and weights.
    for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
    {
        float* weightSumsl = weightSums[lvl];
        float* idepthl = idepth[lvl];
        Eigen::Vector3f* dIRefl = lastRef->dIp[lvl];

        int wl = w[lvl], hl = h[lvl];

        int lpc_n=0;
        float* lpc_u = pc_u[lvl];
        float* lpc_v = pc_v[lvl];
        float* lpc_idepth = pc_idepth[lvl];
        float* lpc_color = pc_color[lvl];


        for(int y=2;y<hl-2;y++)
            for(int x=2;x<wl-2;x++)
            {
                int i = x+y*wl;

                if(weightSumsl[i] > 0)
                {
                    idepthl[i] /= weightSumsl[i];
                    lpc_u[lpc_n] = x;
                    lpc_v[lpc_n] = y;
                    lpc_idepth[lpc_n] = idepthl[i];
                    lpc_color[lpc_n] = dIRefl[i][0];



                    if(!std::isfinite(lpc_color[lpc_n]) || !(idepthl[i]>0))
                    {
                        idepthl[i] = -1;
                        continue;	// just skip if something is wrong.
                    }
                    lpc_n++;
                }
                else
                    idepthl[i] = -1;

                weightSumsl[i] = 1;
            }

        pc_n[lvl] = lpc_n;
//		printf("pc_n[lvl] is %d \n", lpc_n);
    }

}


void CoarseTracker::makeCoarseDepthL0(std::vector<FrameHessian*> frameHessians, FrameHessian* fh_right, CalibHessian Hcalib)
{
	// make coarse tracking templates for latstRef.
	memset(idepth[0], 0, sizeof(float)*w[0]*h[0]);
	memset(weightSums[0], 0, sizeof(float)*w[0]*h[0]);
	FrameHessian* fh_target = frameHessians.back();
	for(FrameHessian* fh : frameHessians)
	{
		for(PointHessian* ph : fh->pointHessians)
		{
			if(ph->lastResiduals[0].first != 0 && ph->lastResiduals[0].second == ResState::IN)
			{
				PointFrameResidual* r = ph->lastResiduals[0].first;
				assert(r->efResidual->isActive() && r->target == lastRef);
				int u = r->centerProjectedTo[0] + 0.5f;
				int v = r->centerProjectedTo[1] + 0.5f;
				float new_idepth = r->centerProjectedTo[2];
				float weight = sqrtf(1e-3 / (ph->efPoint->HdiF+1e-12));

				idepth[0][u+w[0]*v] += new_idepth *weight;
				weightSums[0][u+w[0]*v] += weight;
			}
		}
	}
// 	for(FrameHessian* fh : frameHessians)
// 	{
// 		for(PointHessian* ph : fh->pointHessians)
// 		{
// 			if(ph->lastResiduals[0].first != 0 && ph->lastResiduals[0].second == ResState::IN) //contains information about residuals to the last two (!) frames. ([0] = latest, [1] = the one before).
// 			{
// 				PointFrameResidual* r = ph->lastResiduals[0].first;
// 				assert(r->efResidual->isActive() && r->target == lastRef);
// 				int u = r->centerProjectedTo[0] + 0.5f;
// 				int v = r->centerProjectedTo[1] + 0.5f;
// 
// 				ImmaturePoint* pt_track = new ImmaturePoint((float)u, (float)v, fh_target, &Hcalib);
// 
// 				pt_track->u_stereo = pt_track->u;
// 				pt_track->v_stereo = pt_track->v;
// 
// 						// free to debug
// 				pt_track->idepth_min_stereo = r->centerProjectedTo[2] * 0.1f;
// 				pt_track->idepth_max_stereo = r->centerProjectedTo[2] * 1.9f;
// 
// 				ImmaturePointStatus pt_track_right = pt_track->traceStereo(fh_right, &Hcalib, 1);
// 
// 				float new_idepth = 0;
// 
// 				if (pt_track_right == ImmaturePointStatus::IPS_GOOD)
// 				{
// 				    ImmaturePoint* pt_track_back = new ImmaturePoint(pt_track->lastTraceUV(0), pt_track->lastTraceUV(1), fh_right, &Hcalib);
// 				    pt_track_back->u_stereo = pt_track_back->u;
// 				    pt_track_back->v_stereo = pt_track_back->v;
// 
// 
// 				    pt_track_back->idepth_min_stereo = r->centerProjectedTo[2] * 0.1f;
// 				    pt_track_back->idepth_max_stereo = r->centerProjectedTo[2] * 1.9f;
// 
// 				    ImmaturePointStatus pt_track_left = pt_track_back->traceStereo(fh_target, &Hcalib, 0);
// 
// 				    float depth = 1.0f/pt_track->idepth_stereo;
// 				    float u_delta = abs(pt_track->u - pt_track_back->lastTraceUV(0));
// 				    if(u_delta<1 && depth > 0 && depth < 50)
// 				    {
// 					new_idepth = pt_track->idepth_stereo;
// 					delete pt_track;
// 					delete pt_track_back;
// 
// 				    } else{
// 
// 					new_idepth = r->centerProjectedTo[2];
// 					delete pt_track;
// 					delete pt_track_back;
// 				    }
// 
// 				}else{
// 
// 				    new_idepth = r->centerProjectedTo[2];
// 				    delete pt_track;
// 
// 				}
// 
// 				float weight = sqrtf(1e-3 / (ph->efPoint->HdiF+1e-12));
// 
// 				idepth[0][u+w[0]*v] += new_idepth *weight;
// 				weightSums[0][u+w[0]*v] += weight;
// 
// 			}
// 		}
// 	}

	for(int lvl=1; lvl<pyrLevelsUsed; lvl++)
	{
		int lvlm1 = lvl-1;
		int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

		float* idepth_l = idepth[lvl];
		float* weightSums_l = weightSums[lvl];

		float* idepth_lm = idepth[lvlm1];
		float* weightSums_lm = weightSums[lvlm1];

		for(int y=0;y<hl;y++)
			for(int x=0;x<wl;x++)
			{
				int bidx = 2*x   + 2*y*wlm1;
				idepth_l[x + y*wl] = 		idepth_lm[bidx] +
											idepth_lm[bidx+1] +
											idepth_lm[bidx+wlm1] +
											idepth_lm[bidx+wlm1+1];

				weightSums_l[x + y*wl] = 	weightSums_lm[bidx] +
											weightSums_lm[bidx+1] +
											weightSums_lm[bidx+wlm1] +
											weightSums_lm[bidx+wlm1+1];
			}
	}


    // dilate idepth by 1.
	for(int lvl=0; lvl<2; lvl++)
	{
		int numIts = 1;


		for(int it=0;it<numIts;it++)
		{
			int wh = w[lvl]*h[lvl]-w[lvl];
			int wl = w[lvl];
			float* weightSumsl = weightSums[lvl];
			float* weightSumsl_bak = weightSums_bak[lvl];
			memcpy(weightSumsl_bak, weightSumsl, w[lvl]*h[lvl]*sizeof(float));
			float* idepthl = idepth[lvl];	// dotnt need to make a temp copy of depth, since I only
											// read values with weightSumsl>0, and write ones with weightSumsl<=0.
			for(int i=w[lvl];i<wh;i++)
			{
				if(weightSumsl_bak[i] <= 0)
				{
					float sum=0, num=0, numn=0;
					if(weightSumsl_bak[i+1+wl] > 0) { sum += idepthl[i+1+wl]; num+=weightSumsl_bak[i+1+wl]; numn++;}
					if(weightSumsl_bak[i-1-wl] > 0) { sum += idepthl[i-1-wl]; num+=weightSumsl_bak[i-1-wl]; numn++;}
					if(weightSumsl_bak[i+wl-1] > 0) { sum += idepthl[i+wl-1]; num+=weightSumsl_bak[i+wl-1]; numn++;}
					if(weightSumsl_bak[i-wl+1] > 0) { sum += idepthl[i-wl+1]; num+=weightSumsl_bak[i-wl+1]; numn++;}
					if(numn>0) {idepthl[i] = sum/numn; weightSumsl[i] = num/numn;}
				}
			}
		}
	}


	// dilate idepth by 1 (2 on lower levels).
	for(int lvl=2; lvl<pyrLevelsUsed; lvl++)
	{
		int wh = w[lvl]*h[lvl]-w[lvl];
		int wl = w[lvl];
		float* weightSumsl = weightSums[lvl];
		float* weightSumsl_bak = weightSums_bak[lvl];
		memcpy(weightSumsl_bak, weightSumsl, w[lvl]*h[lvl]*sizeof(float));
		float* idepthl = idepth[lvl];	// dotnt need to make a temp copy of depth, since I only
										// read values with weightSumsl>0, and write ones with weightSumsl<=0.
		for(int i=w[lvl];i<wh;i++)
		{
			if(weightSumsl_bak[i] <= 0)
			{
				float sum=0, num=0, numn=0;
				if(weightSumsl_bak[i+1] > 0) { sum += idepthl[i+1]; num+=weightSumsl_bak[i+1]; numn++;}
				if(weightSumsl_bak[i-1] > 0) { sum += idepthl[i-1]; num+=weightSumsl_bak[i-1]; numn++;}
				if(weightSumsl_bak[i+wl] > 0) { sum += idepthl[i+wl]; num+=weightSumsl_bak[i+wl]; numn++;}
				if(weightSumsl_bak[i-wl] > 0) { sum += idepthl[i-wl]; num+=weightSumsl_bak[i-wl]; numn++;}
				if(numn>0) {idepthl[i] = sum/numn; weightSumsl[i] = num/numn;}
			}
		}
	}


	// normalize idepths and weights.
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		float* weightSumsl = weightSums[lvl];
		float* idepthl = idepth[lvl];
		Eigen::Vector3f* dIRefl = lastRef->dIp[lvl];

		int wl = w[lvl], hl = h[lvl];

		int lpc_n=0;
		float* lpc_u = pc_u[lvl];
		float* lpc_v = pc_v[lvl];
		float* lpc_idepth = pc_idepth[lvl];
		float* lpc_color = pc_color[lvl];


		for(int y=2;y<hl-2;y++)
			for(int x=2;x<wl-2;x++)
			{
				int i = x+y*wl;

				if(weightSumsl[i] > 0)
				{
					idepthl[i] /= weightSumsl[i];
					lpc_u[lpc_n] = x;
					lpc_v[lpc_n] = y;
					lpc_idepth[lpc_n] = idepthl[i];
					lpc_color[lpc_n] = dIRefl[i][0];



					if(!std::isfinite(lpc_color[lpc_n]) || !(idepthl[i]>0))
					{
						idepthl[i] = -1;
						continue;	// just skip if something is wrong.
					}
					lpc_n++;
				}
				else
					idepthl[i] = -1;

				weightSumsl[i] = 1;
			}

		pc_n[lvl] = lpc_n;
	}
	
}



void CoarseTracker::calcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l)
{
	acc.initialize();

	__m128 fxl = _mm_set1_ps(fx[lvl]);
	__m128 fyl = _mm_set1_ps(fy[lvl]);
	__m128 b0 = _mm_set1_ps(lastRef_aff_g2l.b);
	__m128 a = _mm_set1_ps((float)(AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l)[0]));

	__m128 one = _mm_set1_ps(1);
	__m128 minusOne = _mm_set1_ps(-1);
	__m128 zero = _mm_set1_ps(0);
	
	int n = buf_warped_n;
	assert(n%4==0);
	for(int i=0;i<n;i+=4)
	{
		__m128 dx = _mm_mul_ps(_mm_load_ps(buf_warped_dx+i), fxl);
		__m128 dy = _mm_mul_ps(_mm_load_ps(buf_warped_dy+i), fyl);
		__m128 u = _mm_load_ps(buf_warped_u+i);
		__m128 v = _mm_load_ps(buf_warped_v+i);
		__m128 id = _mm_load_ps(buf_warped_idepth+i);


		acc.updateSSE_eighted(
				_mm_mul_ps(id,dx),
				_mm_mul_ps(id,dy),
				_mm_sub_ps(zero, _mm_mul_ps(id,_mm_add_ps(_mm_mul_ps(u,dx), _mm_mul_ps(v,dy)))),
				_mm_sub_ps(zero, _mm_add_ps(
						_mm_mul_ps(_mm_mul_ps(u,v),dx),
						_mm_mul_ps(dy,_mm_add_ps(one, _mm_mul_ps(v,v))))),
				_mm_add_ps(
						_mm_mul_ps(_mm_mul_ps(u,v),dy),
						_mm_mul_ps(dx,_mm_add_ps(one, _mm_mul_ps(u,u)))),
				_mm_sub_ps(_mm_mul_ps(u,dy), _mm_mul_ps(v,dx)),
				_mm_mul_ps(a,_mm_sub_ps(b0, _mm_load_ps(buf_warped_refColor+i))),
				minusOne,
				_mm_load_ps(buf_warped_residual+i),
				_mm_load_ps(buf_warped_weight+i));
	}

	acc.finish();
	H_out = acc.H.topLeftCorner<8,8>().cast<double>() * (1.0f/n);
	b_out = acc.H.topRightCorner<8,1>().cast<double>() * (1.0f/n);
	
	H_out.block<8,3>(0,0) *= SCALE_XI_TRANS;
	H_out.block<8,3>(0,3) *= SCALE_XI_ROT;
	H_out.block<8,1>(0,6) *= SCALE_A;
	H_out.block<8,1>(0,7) *= SCALE_B;
	H_out.block<3,8>(0,0) *= SCALE_XI_TRANS;
	H_out.block<3,8>(3,0) *= SCALE_XI_ROT;
	H_out.block<1,8>(6,0) *= SCALE_A;
	H_out.block<1,8>(7,0) *= SCALE_B;
	b_out.segment<3>(0) *= SCALE_XI_TRANS;
	b_out.segment<3>(3) *= SCALE_XI_ROT;
	b_out.segment<1>(6) *= SCALE_A;
	b_out.segment<1>(7) *= SCALE_B;
}




Vec6 CoarseTracker::calcRes(int lvl, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH)
{
	float E = 0;
	int numTermsInE = 0;
	int numTermsInWarped = 0;
	int numSaturated=0;

	int wl = w[lvl];
	int hl = h[lvl];
	Eigen::Vector3f* dINewl = newFrame->dIp[lvl];
	float fxl = fx[lvl];
	float fyl = fy[lvl];
	float cxl = cx[lvl];
	float cyl = cy[lvl];


	Mat33f RKi = (refToNew.rotationMatrix().cast<float>() * Ki[lvl]);
	Vec3f t = (refToNew.translation()).cast<float>();
	Vec2f affLL = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l).cast<float>();


	float sumSquaredShiftT=0;
	float sumSquaredShiftRT=0;
	float sumSquaredShiftNum=0;

	float maxEnergy = 2*setting_huberTH*cutoffTH-setting_huberTH*setting_huberTH;	// energy for r=setting_coarseCutoffTH.


    MinimalImageB3* resImage = 0;
	if(debugPlot)
	{
		resImage = new MinimalImageB3(wl,hl);
		resImage->setConst(Vec3b(255,255,255));
	}

	int nl = pc_n[lvl];
	float* lpc_u = pc_u[lvl];
	float* lpc_v = pc_v[lvl];
	float* lpc_idepth = pc_idepth[lvl];
	float* lpc_color = pc_color[lvl];
	
	for(int i=0;i<nl;i++)
	{
		float id = lpc_idepth[i];
		float x = lpc_u[i];
		float y = lpc_v[i];

		Vec3f pt = RKi * Vec3f(x, y, 1) + t*id;
		float u = pt[0] / pt[2];
		float v = pt[1] / pt[2];
		float Ku = fxl * u + cxl;
		float Kv = fyl * v + cyl;
		float new_idepth = id/pt[2];

		if(lvl==0 && i%32==0)
		{
			// translation only (positive)
			Vec3f ptT = Ki[lvl] * Vec3f(x, y, 1) + t*id;
			float uT = ptT[0] / ptT[2];
			float vT = ptT[1] / ptT[2];
			float KuT = fxl * uT + cxl;
			float KvT = fyl * vT + cyl;

			// translation only (negative)
			Vec3f ptT2 = Ki[lvl] * Vec3f(x, y, 1) - t*id;
			float uT2 = ptT2[0] / ptT2[2];
			float vT2 = ptT2[1] / ptT2[2];
			float KuT2 = fxl * uT2 + cxl;
			float KvT2 = fyl * vT2 + cyl;

			//translation and rotation (negative)
			Vec3f pt3 = RKi * Vec3f(x, y, 1) - t*id;
			float u3 = pt3[0] / pt3[2];
			float v3 = pt3[1] / pt3[2];
			float Ku3 = fxl * u3 + cxl;
			float Kv3 = fyl * v3 + cyl;

			//translation and rotation (positive)
			//already have it.

			sumSquaredShiftT += (KuT-x)*(KuT-x) + (KvT-y)*(KvT-y);
			sumSquaredShiftT += (KuT2-x)*(KuT2-x) + (KvT2-y)*(KvT2-y);
			sumSquaredShiftRT += (Ku-x)*(Ku-x) + (Kv-y)*(Kv-y);
			sumSquaredShiftRT += (Ku3-x)*(Ku3-x) + (Kv3-y)*(Kv3-y);
			sumSquaredShiftNum+=2;
		}

		if(!(Ku > 2 && Kv > 2 && Ku < wl-3 && Kv < hl-3 && new_idepth > 0)) continue;



		float refColor = lpc_color[i];
        Vec3f hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl);
        if(!std::isfinite((float)hitColor[0])) continue;
        float residual = hitColor[0] - (float)(affLL[0] * refColor + affLL[1]);
        float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);


		if(fabs(residual) > cutoffTH)
		{
			if(debugPlot) resImage->setPixel4(lpc_u[i], lpc_v[i], Vec3b(0,0,255));
			E += maxEnergy;
			numTermsInE++;
			numSaturated++;
		}
		else
		{
			if(debugPlot) resImage->setPixel4(lpc_u[i], lpc_v[i], Vec3b(residual+128,residual+128,residual+128));

			E += hw *residual*residual*(2-hw);
			numTermsInE++;

			buf_warped_idepth[numTermsInWarped] = new_idepth;
			buf_warped_u[numTermsInWarped] = u;
			buf_warped_v[numTermsInWarped] = v;
			buf_warped_dx[numTermsInWarped] = hitColor[1];
			buf_warped_dy[numTermsInWarped] = hitColor[2];
			buf_warped_residual[numTermsInWarped] = residual;
			buf_warped_weight[numTermsInWarped] = hw;
			buf_warped_refColor[numTermsInWarped] = lpc_color[i];
			numTermsInWarped++;
		}
	}

	while(numTermsInWarped%4!=0)
	{
		buf_warped_idepth[numTermsInWarped] = 0;
		buf_warped_u[numTermsInWarped] = 0;
		buf_warped_v[numTermsInWarped] = 0;
		buf_warped_dx[numTermsInWarped] = 0;
		buf_warped_dy[numTermsInWarped] = 0;
		buf_warped_residual[numTermsInWarped] = 0;
		buf_warped_weight[numTermsInWarped] = 0;
		buf_warped_refColor[numTermsInWarped] = 0;
		numTermsInWarped++;
	}
	buf_warped_n = numTermsInWarped;

	if(debugPlot)
	{
		IOWrap::displayImage("RES", resImage, false);
		IOWrap::waitKey(0);
		delete resImage;
	}

	Vec6 rs;
	rs[0] = E;
	rs[1] = numTermsInE;
	rs[2] = sumSquaredShiftT/(sumSquaredShiftNum+0.1);
	rs[3] = 0;
	rs[4] = sumSquaredShiftRT/(sumSquaredShiftNum+0.1);
	rs[5] = numSaturated / (float)numTermsInE;

// 	LOG(INFO)<<"E: "<<E;
// 	LOG(INFO)<<"numTermsInE: "<<numTermsInE;
// 	exit(1);
	return rs;
}


void CoarseTracker::setCTRefForFirstFrame(std::vector<FrameHessian *> frameHessians)
{
    assert(frameHessians.size()>0);
    lastRef = frameHessians.back();

    makeCoarseDepthForFirstFrame(lastRef);

    refFrameID = lastRef->shell->id;
    lastRef_aff_g2l = lastRef->aff_g2l();

    firstCoarseRMSE=-1;
}



void CoarseTracker::setCoarseTrackingRef(
		std::vector<FrameHessian*> frameHessians, FrameHessian* fh_right, CalibHessian Hcalib)
{
	assert(frameHessians.size()>0);
	lastRef = frameHessians.back();
	makeCoarseDepthL0(frameHessians, fh_right, Hcalib);



	refFrameID = lastRef->shell->id;
	lastRef_aff_g2l = lastRef->aff_g2l();

	firstCoarseRMSE=-1;

}
bool CoarseTracker::trackNewestCoarse(
		FrameHessian* newFrameHessian,
		SE3 &lastToNew_out, AffLight &aff_g2l_out,
		int coarsestLvl,
		Vec5 minResForAbort,
		IOWrap::Output3DWrapper* wrap)
{
	debugPlot = setting_render_displayCoarseTrackingFull;
	debugPrint = false;

	assert(coarsestLvl < 5 && coarsestLvl < pyrLevelsUsed);

	lastResiduals.setConstant(NAN);
	lastFlowIndicators.setConstant(1000);


	newFrame = newFrameHessian;
	int maxIterations[] = {10,20,50,50,50};
	float lambdaExtrapolationLimit = 0.001;

	SE3 refToNew_current = lastToNew_out;
	AffLight aff_g2l_current = aff_g2l_out;

	bool haveRepeated = false;

	IMUPreintegrator IMU_preintegrator;
	double time_start = pic_time_stamp[lastRef->shell->incoming_id];
	double time_end = pic_time_stamp[newFrame->shell->incoming_id];
// 	LOG(INFO)<<"lastRef->shell->incoming_id: "<<lastRef->shell->incoming_id<<" newFrame->shell->incoming_id: "<<newFrame->shell->incoming_id;
	
	int index;
// 	LOG(INFO)<<"pic_time_stamp.size(): "<<pic_time_stamp.size();
// 	LOG(INFO)<<std::fixed<<std::setprecision(9)<<"time_start: "<<time_start<<" time_end: "<<time_start<<" dt: "<<time_end - time_start;
	for(int i=0;i<imu_time_stamp.size();++i){
	    if(imu_time_stamp[i]>time_start||fabs(time_start-imu_time_stamp[i])<0.001){
		index = i;
		break;
	    }
	}
	
	while(1){
	    double delta_t; 
	    if(imu_time_stamp[index+1]<time_end)
	      delta_t = imu_time_stamp[index+1]-imu_time_stamp[index];
	    else{
	      delta_t = time_end - imu_time_stamp[index];
	      if(delta_t<0.000001)break;
	    }
	    IMU_preintegrator.update(m_gry[index]-lastRef->bias_g, m_acc[index]-lastRef->bias_a, delta_t);
	    if(imu_time_stamp[index+1]>=time_end)
	      break;
	    index++;
	}
	
	std::vector<double> imu_track_w(coarsestLvl,0);
	imu_track_w[0] = imu_weight_tracker;
	imu_track_w[1] = imu_track_w[0]/1.2;
	imu_track_w[2] = imu_track_w[1]/1.5;
	imu_track_w[3] = imu_track_w[2]/2;
	imu_track_w[4] = imu_track_w[3]/3;
	
	for(int lvl=coarsestLvl; lvl>=0; lvl--)
	{
		Mat88 H; Vec8 b;
		float levelCutoffRepeat=1;
		Vec6 resOld = calcRes(lvl, refToNew_current, aff_g2l_current, setting_coarseCutoffTH*levelCutoffRepeat);
		while(resOld[5] > 0.6 && levelCutoffRepeat < 50)
		{
			levelCutoffRepeat*=2;
			resOld = calcRes(lvl, refToNew_current, aff_g2l_current, setting_coarseCutoffTH*levelCutoffRepeat);

            if(!setting_debugout_runquiet)
                printf("INCREASING cutoff to %f (ratio is %f)!\n", setting_coarseCutoffTH*levelCutoffRepeat, resOld[5]);
		}
		
		calcGSSSE(lvl, H, b, refToNew_current, aff_g2l_current);
		Mat66 H_imu;
		Vec6 b_imu;
		Vec9 res_PVPhi;
		double res_imu_old = 0;
		if(lvl<=0){
		    res_imu_old = calcIMUResAndGS(H_imu, b_imu, refToNew_current, IMU_preintegrator,res_PVPhi,resOld[0],imu_track_w[lvl]);
// 		    LOG(INFO)<<"res_imu_old: "<<res_imu_old<<" resOld[0]: "<<resOld[0]<<" resOld[1]: "<<resOld[0];
		}
	    
		float lambda = 0.01;

		if(debugPrint)
		{
			Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l_current).cast<float>();
			printf("lvl%d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = %f)! \t",
					lvl, -1, lambda, 1.0f,
					"INITIA",
					0.0f,
					resOld[0] / resOld[1],
					 0,(int)resOld[1],
					0.0f);
			std::cout << refToNew_current.log().transpose() << " AFF " << aff_g2l_current.vec().transpose() <<" (rel " << relAff.transpose() << ")\n";
		}

		for(int iteration=0; iteration < maxIterations[lvl]; iteration++)
		{
			Mat88 Hl = H;
// 			Hl = Mat88::Zero();
// 			b = Vec8::Zero();
// 			if(lvl==0){
// 			    LOG(INFO)<<"H_image: \n"<<H;
// 			    LOG(INFO)<<"b_image: "<<b.transpose();
// 			}
			if(imu_use_flag&&imu_track_flag&&imu_track_ready&&lvl<=0){
			  Hl.block(0,0,6,6) = Hl.block(0,0,6,6) + H_imu;
			  b.block(0,0,6,1) = b.block(0,0,6,1) + b_imu.block(0,0,6,1);
			}
			
			for(int i=0;i<8;i++) Hl(i,i) *= (1+lambda);
			
			Vec8 inc = Hl.ldlt().solve(-b);

			if(setting_affineOptModeA < 0 && setting_affineOptModeB < 0)	// fix a, b
			{
				inc.head<6>() = Hl.topLeftCorner<6,6>().ldlt().solve(-b.head<6>());
			 	inc.tail<2>().setZero();
			}
			if(!(setting_affineOptModeA < 0) && setting_affineOptModeB < 0)	// fix b
			{
				inc.head<7>() = Hl.topLeftCorner<7,7>().ldlt().solve(-b.head<7>());
			 	inc.tail<1>().setZero();
			}
			if(setting_affineOptModeA < 0 && !(setting_affineOptModeB < 0))	// fix a
			{
				Mat88 HlStitch = Hl;
				Vec8 bStitch = b;
				HlStitch.col(6) = HlStitch.col(7);
				HlStitch.row(6) = HlStitch.row(7);
				bStitch[6] = bStitch[7];
				Vec7 incStitch = HlStitch.topLeftCorner<7,7>().ldlt().solve(-bStitch.head<7>());
				inc.setZero();
				inc.head<6>() = incStitch.head<6>();
				inc[6] = 0;
				inc[7] = incStitch[6];
			}




			float extrapFac = 1;
			if(lambda < lambdaExtrapolationLimit) extrapFac = sqrt(sqrt(lambdaExtrapolationLimit / lambda));
			inc *= extrapFac;

			Vec8 incScaled = inc;
// 			incScaled.segment<3>(0) *= SCALE_XI_ROT;
// 			incScaled.segment<3>(3) *= SCALE_XI_TRANS;
// 			incScaled.segment<1>(6) *= SCALE_A;
// 			incScaled.segment<1>(7) *= SCALE_B;
			incScaled.segment<3>(0) *= SCALE_XI_TRANS;
			incScaled.segment<3>(3) *= SCALE_XI_ROT;
			incScaled.segment<1>(6) *= SCALE_A;
			incScaled.segment<1>(7) *= SCALE_B;

            if(!std::isfinite(incScaled.sum())) incScaled.setZero();
// 			LOG(INFO)<<"incScaled: "<<incScaled.transpose();s

			SE3 refToNew_new = SE3::exp((Vec6)(incScaled.head<6>())) * refToNew_current;
			AffLight aff_g2l_new = aff_g2l_current;
			aff_g2l_new.a += incScaled[6];
			aff_g2l_new.b += incScaled[7];

			Vec6 resNew = calcRes(lvl, refToNew_new, aff_g2l_new, setting_coarseCutoffTH*levelCutoffRepeat);
			double res_imu_new;
			if(lvl<=0){
			  res_imu_new = calcIMUResAndGS(H_imu, b_imu, refToNew_new, IMU_preintegrator,res_PVPhi,resNew[0],imu_track_w[lvl]);
			}

			bool accept = (resNew[0] / resNew[1]) < (resOld[0] / resOld[1]);
			if(imu_use_flag&&imu_track_flag&&imu_track_ready&&lvl<=0){
			  accept = (resNew[0] / resNew[1] * resOld[1] + res_imu_new) < (resOld[0] + res_imu_old);
			}

			if(debugPrint)
			{
				Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l_new).cast<float>();
				printf("lvl %d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = %f)! \t",
						lvl, iteration, lambda,
						extrapFac,
						(accept ? "ACCEPT" : "REJECT"),
						resOld[0] / resOld[1],
						resNew[0] / resNew[1],
						(int)resOld[1], (int)resNew[1],
						inc.norm());
				std::cout << refToNew_new.log().transpose() << " AFF " << aff_g2l_new.vec().transpose() <<" (rel " << relAff.transpose() << ")\n";
			}
			if(accept)
			{
				calcGSSSE(lvl, H, b, refToNew_new, aff_g2l_new);
				resOld = resNew;
				res_imu_old = res_imu_new;
				aff_g2l_current = aff_g2l_new;
				refToNew_current = refToNew_new;
				lambda *= 0.5;
			}
			else
			{
				lambda *= 4;
				if(lambda < lambdaExtrapolationLimit) lambda = lambdaExtrapolationLimit;
			}

			if(!(inc.norm() > 1e-3))
			{
				if(debugPrint)
					printf("inc too small, break!\n");
				break;
			}
		}

		// set last residual for that level, as well as flow indicators.
		lastResiduals[lvl] = sqrtf((float)(resOld[0] / resOld[1]));
		lastFlowIndicators = resOld.segment<3>(2);
		if(lastResiduals[lvl] > 1.5*minResForAbort[lvl]) return false;


		if(levelCutoffRepeat > 1 && !haveRepeated)
		{
			lvl++;
			haveRepeated=true;
			printf("REPEAT LEVEL!\n");
		}
	}

	// set!
	lastToNew_out = refToNew_current;
	aff_g2l_out = aff_g2l_current;


	if((setting_affineOptModeA != 0 && (fabsf(aff_g2l_out.a) > 1.2))
	|| (setting_affineOptModeB != 0 && (fabsf(aff_g2l_out.b) > 200)))
		return false;

	Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l_out).cast<float>();

	if((setting_affineOptModeA == 0 && (fabsf(logf((float)relAff[0])) > 1.5))
	|| (setting_affineOptModeB == 0 && (fabsf((float)relAff[1]) > 200)))
		return false;



	if(setting_affineOptModeA < 0) aff_g2l_out.a=0;
	if(setting_affineOptModeB < 0) aff_g2l_out.b=0;

	return true;
}



void CoarseTracker::debugPlotIDepthMap(float* minID_pt, float* maxID_pt, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
    if(w[1] == 0) return;


	int lvl = 0;

	{
		std::vector<float> allID;
		for(int i=0;i<h[lvl]*w[lvl];i++)
		{
			if(idepth[lvl][i] > 0)
				allID.push_back(idepth[lvl][i]);
		}
		std::sort(allID.begin(), allID.end());
		int n = allID.size()-1;

		float minID_new = allID[(int)(n*0.05)];
		float maxID_new = allID[(int)(n*0.95)];

		float minID, maxID;
		minID = minID_new;
		maxID = maxID_new;
		if(minID_pt!=0 && maxID_pt!=0)
		{
			if(*minID_pt < 0 || *maxID_pt < 0)
			{
				*maxID_pt = maxID;
				*minID_pt = minID;
			}
			else
			{

				// slowly adapt: change by maximum 10% of old span.
				float maxChange = 0.3*(*maxID_pt - *minID_pt);

				if(minID < *minID_pt - maxChange)
					minID = *minID_pt - maxChange;
				if(minID > *minID_pt + maxChange)
					minID = *minID_pt + maxChange;


				if(maxID < *maxID_pt - maxChange)
					maxID = *maxID_pt - maxChange;
				if(maxID > *maxID_pt + maxChange)
					maxID = *maxID_pt + maxChange;

				*maxID_pt = maxID;
				*minID_pt = minID;
			}
		}


		MinimalImageB3 mf(w[lvl], h[lvl]);
		mf.setBlack();
		for(int i=0;i<h[lvl]*w[lvl];i++)
		{
			int c = lastRef->dIp[lvl][i][0]*0.9f;
			if(c>255) c=255;
			mf.at(i) = Vec3b(c,c,c);
		}
		int wl = w[lvl];
		for(int y=3;y<h[lvl]-3;y++)
			for(int x=3;x<wl-3;x++)
			{
				int idx=x+y*wl;
				float sid=0, nid=0;
				float* bp = idepth[lvl]+idx;

				if(bp[0] > 0) {sid+=bp[0]; nid++;}
				if(bp[1] > 0) {sid+=bp[1]; nid++;}
				if(bp[-1] > 0) {sid+=bp[-1]; nid++;}
				if(bp[wl] > 0) {sid+=bp[wl]; nid++;}
				if(bp[-wl] > 0) {sid+=bp[-wl]; nid++;}

				if(bp[0] > 0 || nid >= 3)
				{
					float id = ((sid / nid)-minID) / ((maxID-minID));
					mf.setPixelCirc(x,y,makeJet3B(id));
					//mf.at(idx) = makeJet3B(id);
				}
			}
        //IOWrap::displayImage("coarseDepth LVL0", &mf, false);


        for(IOWrap::Output3DWrapper* ow : wraps)
            ow->pushDepthImage(&mf);

		if(debugSaveImages)
		{
			char buf[1000];
			snprintf(buf, 1000, "images_out/predicted_%05d_%05d.png", lastRef->shell->id, refFrameID);
			IOWrap::writeImage(buf,&mf);
		}

	}
}



void CoarseTracker::debugPlotIDepthMapFloat(std::vector<IOWrap::Output3DWrapper*> &wraps)
{
    if(w[1] == 0) return;
    int lvl = 0;
    MinimalImageF mim(w[lvl], h[lvl], idepth[lvl]);
    for(IOWrap::Output3DWrapper* ow : wraps)
        ow->pushDepthImageFloat(&mim, lastRef);
}











CoarseDistanceMap::CoarseDistanceMap(int ww, int hh)
{
	fwdWarpedIDDistFinal = new float[ww*hh/4];

	bfsList1 = new Eigen::Vector2i[ww*hh/4];
	bfsList2 = new Eigen::Vector2i[ww*hh/4];

	int fac = 1 << (pyrLevelsUsed-1);


	coarseProjectionGrid = new PointFrameResidual*[2048*(ww*hh/(fac*fac))];
	coarseProjectionGridNum = new int[ww*hh/(fac*fac)];

	w[0]=h[0]=0;
}
CoarseDistanceMap::~CoarseDistanceMap()
{
	delete[] fwdWarpedIDDistFinal;
	delete[] bfsList1;
	delete[] bfsList2;
	delete[] coarseProjectionGrid;
	delete[] coarseProjectionGridNum;
}





void CoarseDistanceMap::makeDistanceMap(
		std::vector<FrameHessian*> frameHessians,
		FrameHessian* frame)
{
	int w1 = w[1];
	int h1 = h[1];
	int wh1 = w1*h1;
	for(int i=0;i<wh1;i++)
		fwdWarpedIDDistFinal[i] = 1000;


	// make coarse tracking templates for latstRef.
	int numItems = 0;

	for(FrameHessian* fh : frameHessians)
	{
		if(frame == fh) continue;

		SE3 fhToNew = frame->PRE_worldToCam * fh->PRE_camToWorld;
		Mat33f KRKi = (K[1] * fhToNew.rotationMatrix().cast<float>() * Ki[0]);
		Vec3f Kt = (K[1] * fhToNew.translation().cast<float>());

		for(PointHessian* ph : fh->pointHessians)
		{
			assert(ph->status == PointHessian::ACTIVE);
			Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt*ph->idepth_scaled;
			int u = ptp[0] / ptp[2] + 0.5f;
			int v = ptp[1] / ptp[2] + 0.5f;
			if(!(u > 0 && v > 0 && u < w[1] && v < h[1])) continue;
			fwdWarpedIDDistFinal[u+w1*v]=0;
			bfsList1[numItems] = Eigen::Vector2i(u,v);
			numItems++;
		}
	}

	growDistBFS(numItems);
}




void CoarseDistanceMap::makeInlierVotes(std::vector<FrameHessian*> frameHessians)
{

}



void CoarseDistanceMap::growDistBFS(int bfsNum)
{
	assert(w[0] != 0);
	int w1 = w[1], h1 = h[1];
	for(int k=1;k<40;k++)
	{
		int bfsNum2 = bfsNum;
		std::swap<Eigen::Vector2i*>(bfsList1,bfsList2);
		bfsNum=0;

		if(k%2==0)
		{
			for(int i=0;i<bfsNum2;i++)
			{
				int x = bfsList2[i][0];
				int y = bfsList2[i][1];
				if(x==0 || y== 0 || x==w1-1 || y==h1-1) continue;
				int idx = x + y * w1;

				if(fwdWarpedIDDistFinal[idx+1] > k)
				{
					fwdWarpedIDDistFinal[idx+1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x+1,y); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-1] > k)
				{
					fwdWarpedIDDistFinal[idx-1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x-1,y); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx+w1] > k)
				{
					fwdWarpedIDDistFinal[idx+w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x,y+1); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-w1] > k)
				{
					fwdWarpedIDDistFinal[idx-w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x,y-1); bfsNum++;
				}
			}
		}
		else
		{
			for(int i=0;i<bfsNum2;i++)
			{
				int x = bfsList2[i][0];
				int y = bfsList2[i][1];
				if(x==0 || y== 0 || x==w1-1 || y==h1-1) continue;
				int idx = x + y * w1;

				if(fwdWarpedIDDistFinal[idx+1] > k)
				{
					fwdWarpedIDDistFinal[idx+1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x+1,y); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-1] > k)
				{
					fwdWarpedIDDistFinal[idx-1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x-1,y); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx+w1] > k)
				{
					fwdWarpedIDDistFinal[idx+w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x,y+1); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-w1] > k)
				{
					fwdWarpedIDDistFinal[idx-w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x,y-1); bfsNum++;
				}

				if(fwdWarpedIDDistFinal[idx+1+w1] > k)
				{
					fwdWarpedIDDistFinal[idx+1+w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x+1,y+1); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-1+w1] > k)
				{
					fwdWarpedIDDistFinal[idx-1+w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x-1,y+1); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-1-w1] > k)
				{
					fwdWarpedIDDistFinal[idx-1-w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x-1,y-1); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx+1-w1] > k)
				{
					fwdWarpedIDDistFinal[idx+1-w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x+1,y-1); bfsNum++;
				}
			}
		}
	}
}


void CoarseDistanceMap::addIntoDistFinal(int u, int v)
{
	if(w[0] == 0) return;
	bfsList1[0] = Eigen::Vector2i(u,v);
	fwdWarpedIDDistFinal[u+w[1]*v] = 0;
	growDistBFS(1);
}



void CoarseDistanceMap::makeK(CalibHessian* HCalib)
{
	w[0] = wG[0];
	h[0] = hG[0];

	fx[0] = HCalib->fxl();
	fy[0] = HCalib->fyl();
	cx[0] = HCalib->cxl();
	cy[0] = HCalib->cyl();

	for (int level = 1; level < pyrLevelsUsed; ++ level)
	{
		w[level] = w[0] >> level;
		h[level] = h[0] >> level;
		fx[level] = fx[level-1] * 0.5;
		fy[level] = fy[level-1] * 0.5;
		cx[level] = (cx[0] + 0.5) / ((int)1<<level) - 0.5;
		cy[level] = (cy[0] + 0.5) / ((int)1<<level) - 0.5;
	}

	for (int level = 0; level < pyrLevelsUsed; ++ level)
	{
		K[level]  << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
		Ki[level] = K[level].inverse();
		fxi[level] = Ki[level](0,0);
		fyi[level] = Ki[level](1,1);
		cxi[level] = Ki[level](0,2);
		cyi[level] = Ki[level](1,2);
	}
}

}
