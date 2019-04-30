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


#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "OptimizationBackend/AccumulatedSCHessian.h"
#include "OptimizationBackend/AccumulatedTopHessian.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{


bool EFAdjointsValid = false;
bool EFIndicesValid = false;
bool EFDeltaValid = false;

void EnergyFunctional::getIMUHessian(MatXX &H, VecX &b){
    H = MatXX::Zero(7+nFrames*15, 7+nFrames*15);
    b = VecX::Zero(7+nFrames*15);
//     if(nFrames ==3)exit(1);
    if(nFrames==1)return;
    int count_imu_res = 0;
    double Energy = 0;
    for(int i=0;i<frames.size()-1;++i){
	MatXX J_all = MatXX::Zero(9, 7+nFrames*15);
	VecX r_all = VecX::Zero(9);
	//preintegrate
	IMUPreintegrator IMU_preintegrator;
	IMU_preintegrator.reset();
	double time_start = pic_time_stamp[frames[i]->data->shell->incoming_id];
	double time_end = pic_time_stamp[frames[i+1]->data->shell->incoming_id];
	double dt = time_end-time_start;
	
// 	if(dt>0.5)continue;
	count_imu_res++;
// 	LOG(INFO)<<"dt: "<<dt;
	FrameHessian* Framei = frames[i]->data;
	FrameHessian* Framej = frames[i+1]->data;
	
	//bias model
	MatXX J_all2 = MatXX::Zero(6, 7+nFrames*15);
	VecX r_all2 = VecX::Zero(6);
	
	r_all2.block(0,0,3,1) = Framej->bias_g+Framej->delta_bias_g - (Framei->bias_g+Framei->delta_bias_g);
	r_all2.block(3,0,3,1) = Framej->bias_a+Framej->delta_bias_a - (Framei->bias_a+Framei->delta_bias_a);
	
	J_all2.block(0,7+i*15+9,3,3) = -Mat33::Identity();
	J_all2.block(0,7+(i+1)*15+9,3,3) = Mat33::Identity();
	J_all2.block(3,7+i*15+12,3,3) = -Mat33::Identity();
	J_all2.block(3,7+(i+1)*15+12,3,3) = Mat33::Identity();
	Mat66 Cov_bias = Mat66::Zero();
	Cov_bias.block(0,0,3,3) = GyrRandomWalkNoise*dt;
	Cov_bias.block(3,3,3,3) = AccRandomWalkNoise*dt;
	Mat66 weight_bias = Mat66::Identity()*imu_weight*imu_weight*Cov_bias.inverse();
// 	weight_bias *= (bei*bei);
	H += J_all2.transpose()*weight_bias*J_all2;
	b += J_all2.transpose()*weight_bias*r_all2;
	
	if(dt>0.5)continue;
	
	SE3 worldToCam_i = Framei->get_worldToCam_evalPT();
	SE3 worldToCam_j = Framej->get_worldToCam_evalPT();
	SE3 worldToCam_i2 = Framei->PRE_worldToCam;
	SE3 worldToCam_j2 = Framej->PRE_worldToCam;

	int index;
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
	    IMU_preintegrator.update(m_gry[index]-Framei->bias_g, m_acc[index]-Framei->bias_a, delta_t);
	    if(imu_time_stamp[index+1]>=time_end)
	      break;
	    index++;
	}
	
	Vec3 g_w;
	g_w << 0,0,-G_norm;
// 	LOG(INFO)<<"00000000";
	
	Mat44 M_WB = T_WD.matrix()*worldToCam_i.inverse().matrix()*T_WD.inverse().matrix()*T_BC.inverse().matrix();
	SE3 T_WB(M_WB);
	Mat33 R_WB = T_WB.rotationMatrix();
	Vec3 t_WB = T_WB.translation();
	
	Mat44 M_WB2 = T_WD.matrix()*worldToCam_i2.inverse().matrix()*T_WD.inverse().matrix()*T_BC.inverse().matrix();
	SE3 T_WB2(M_WB2);
	Mat33 R_WB2 = T_WB2.rotationMatrix();
	Vec3 t_WB2 = T_WB2.translation();
	
	Mat44 M_WBj = T_WD.matrix()*worldToCam_j.inverse().matrix()*T_WD.inverse().matrix()*T_BC.inverse().matrix();
	SE3 T_WBj(M_WBj);
	Mat33 R_WBj = T_WBj.rotationMatrix();
	Vec3 t_WBj = T_WBj.translation();
	
	Mat44 M_WBj2 = T_WD.matrix()*worldToCam_j2.inverse().matrix()*T_WD.inverse().matrix()*T_BC.inverse().matrix();
	SE3 T_WBj2(M_WBj2);
	Mat33 R_WBj2 = T_WBj2.rotationMatrix();
	Vec3 t_WBj2 = T_WBj2.translation();
	
// 	LOG(INFO)<<"a";
	//calculate res
// 	Vec3 so3 = IMU_preintegrator.getJRBiasg()*Framei->delta_bias_g;
	Mat33 R_temp = SO3::exp(IMU_preintegrator.getJRBiasg()*Framei->delta_bias_g).matrix();
// 	Mat33 res_R = (IMU_preintegrator.getDeltaR()*R_temp).transpose()*R_WB.transpose()*R_WBj;
// 	Vec3 res_phi = SO3(res_R).log();
// 	Vec3 res_v = R_WB.transpose()*(Framej->velocity-Framei->velocity-g_w*dt)-
// 		 (IMU_preintegrator.getDeltaV()+IMU_preintegrator.getJVBiasa()*Framei->delta_bias_a+IMU_preintegrator.getJVBiasg()*Framei->delta_bias_g);
// 	Vec3 res_p = R_WB.transpose()*(t_WBj-t_WB-Framei->velocity*dt-0.5*g_w*dt*dt)-
// 		 (IMU_preintegrator.getDeltaP()+IMU_preintegrator.getJPBiasa()*Framei->delta_bias_a+IMU_preintegrator.getJPBiasg()*Framei->delta_bias_g);
		 
	Mat33 res_R2 = (IMU_preintegrator.getDeltaR()*R_temp).transpose()*R_WB2.transpose()*R_WBj2;
	Vec3 res_phi2 = SO3(res_R2).log();
	Vec3 res_v2 = R_WB2.transpose()*(Framej->velocity-Framei->velocity-g_w*dt)-
		 (IMU_preintegrator.getDeltaV()+IMU_preintegrator.getJVBiasa()*Framei->delta_bias_a+IMU_preintegrator.getJVBiasg()*Framei->delta_bias_g);
	Vec3 res_p2 = R_WB2.transpose()*(t_WBj2-t_WB2-Framei->velocity*dt-0.5*g_w*dt*dt)-
		 (IMU_preintegrator.getDeltaP()+IMU_preintegrator.getJPBiasa()*Framei->delta_bias_a+IMU_preintegrator.getJPBiasg()*Framei->delta_bias_g);
// 	LOG(INFO)<<"R_WB2: \n"<<R_WB2;
// 	LOG(INFO)<<"(Framej->velocity-Framei->velocity-g_w*dt): "<<(Framej->velocity-Framei->velocity-g_w*dt).transpose();
// 	if(i==0){
// 	  LOG(INFO)<<"worldToCam_i: \n"<<worldToCam_i.matrix();
// 	  LOG(INFO)<<"res_phi: "<<res_phi2.transpose();
// 	  LOG(INFO)<<"res_v: "<<res_v2.transpose();
// 	  LOG(INFO)<<"res_p: "<<res_p2.transpose();
// 	}
// 	LOG(INFO)<<"i: "<<i<<" T_WD.scale(): "<<T_WD.scale()<<" Framei time: "<<std::fixed<<std::setprecision(14)<<pic_time_stamp[Framei->shell->incoming_id];
// 	LOG(INFO)<<" Framei->velocity: "<<Framei->velocity.transpose()<<" Framej->velocity: "<<Framej->velocity.transpose();
// 	LOG(INFO)<<"res_v2: "<<res_v2.transpose();
// 	LOG(INFO)<<"res_p2: "<<res_p2.transpose();
	Mat99 Cov = IMU_preintegrator.getCovPVPhi();

// 	Mat33 J_resPhi_phi_i = -IMU_preintegrator.JacobianRInv(res_phi2)*R_WBj.transpose()*R_WB;
// 	Mat33 J_resPhi_phi_j = IMU_preintegrator.JacobianRInv(res_phi2);
// 	Mat33 J_resPhi_bg = -IMU_preintegrator.JacobianRInv(res_phi2)*SO3::exp(-res_phi2).matrix()*
// 		 IMU_preintegrator.JacobianR(IMU_preintegrator.getJRBiasg()*Framei->delta_bias_g)*IMU_preintegrator.getJRBiasg();
// 
// 	Mat33 J_resV_phi_i = SO3::hat(R_WB.transpose()*(Framej->velocity - Framei->velocity - g_w*dt));
// 	Mat33 J_resV_v_i = -R_WB.transpose();
// 	Mat33 J_resV_v_j = R_WB.transpose();
// 	Mat33 J_resV_ba = -IMU_preintegrator.getJVBiasa();
// 	Mat33 J_resV_bg = -IMU_preintegrator.getJVBiasg();
// 
// 	Mat33 J_resP_p_i = -Mat33::Identity();	
// 	Mat33 J_resP_p_j = R_WB.transpose()*R_WBj;
// 	Mat33 J_resP_bg = -IMU_preintegrator.getJPBiasg();
// 	Mat33 J_resP_ba = -IMU_preintegrator.getJPBiasa();
// 	Mat33 J_resP_v_i = -R_WB.transpose()*dt;
// 	Mat33 J_resP_phi_i = SO3::hat(R_WB.transpose()*(t_WBj - t_WB - Framei->velocity*dt - 0.5*g_w*dt*dt));
	
	Mat33 J_resPhi_phi_i = -IMU_preintegrator.JacobianRInv(res_phi2)*R_WBj2.transpose()*R_WB2;
	Mat33 J_resPhi_phi_j = IMU_preintegrator.JacobianRInv(res_phi2);
	Mat33 J_resPhi_bg = -IMU_preintegrator.JacobianRInv(res_phi2)*SO3::exp(-res_phi2).matrix()*
		 IMU_preintegrator.JacobianR(IMU_preintegrator.getJRBiasg()*Framei->delta_bias_g)*IMU_preintegrator.getJRBiasg();
// 	LOG(INFO)<<"c";
	Mat33 J_resV_phi_i = SO3::hat(R_WB2.transpose()*(Framej->velocity - Framei->velocity - g_w*dt));
	Mat33 J_resV_v_i = -R_WB2.transpose();
	Mat33 J_resV_v_j = R_WB2.transpose();
	Mat33 J_resV_ba = -IMU_preintegrator.getJVBiasa();
	Mat33 J_resV_bg = -IMU_preintegrator.getJVBiasg();
// 	LOG(INFO)<<"d";
	Mat33 J_resP_p_i = -Mat33::Identity();	
	Mat33 J_resP_p_j = R_WB2.transpose()*R_WBj2;
	Mat33 J_resP_bg = -IMU_preintegrator.getJPBiasg();
	Mat33 J_resP_ba = -IMU_preintegrator.getJPBiasa();
	Mat33 J_resP_v_i = -R_WB2.transpose()*dt;
	Mat33 J_resP_phi_i = SO3::hat(R_WB2.transpose()*(t_WBj2 - t_WB2 - Framei->velocity*dt - 0.5*g_w*dt*dt));
// 	LOG(INFO)<<"1111111";

// 	LOG(INFO)<<"222222";
	Mat915 J_imui = Mat915::Zero();//rho,phi,v,bias_g,bias_a;
	J_imui.block(0,0,3,3) = J_resP_p_i;
	J_imui.block(0,3,3,3) = J_resP_phi_i;
	J_imui.block(0,6,3,3) = J_resP_v_i;
	J_imui.block(0,9,3,3) = J_resP_bg;
	J_imui.block(0,12,3,3) = J_resP_ba;
	
	J_imui.block(3,3,3,3) = J_resPhi_phi_i;
	J_imui.block(3,9,3,3) = J_resPhi_bg;
	
	J_imui.block(6,3,3,3) = J_resV_phi_i;
	J_imui.block(6,6,3,3) = J_resV_v_i;
	J_imui.block(6,9,3,3) = J_resV_bg;
	J_imui.block(6,12,3,3) = J_resV_ba;
	
	
	
	Mat915 J_imuj = Mat915::Zero();
	J_imuj.block(0,0,3,3) = J_resP_p_j;
	J_imuj.block(3,3,3,3) = J_resPhi_phi_j;
	J_imuj.block(6,6,3,3)  = J_resV_v_j;
// 	LOG(INFO)<<"333333";
	Mat99 Weight = Mat99::Zero();
	Weight.block(0,0,3,3) = Cov.block(0,0,3,3);
	Weight.block(3,3,3,3) = Cov.block(6,6,3,3);
        Weight.block(6,6,3,3) = Cov.block(3,3,3,3);
	Mat99 Weight2 = Mat99::Zero();
	for(int i=0;i<9;++i){
	    Weight2(i,i) = Weight(i,i);
	}
	Weight = Weight2;
// 	Mat99 Weight_sqrt = Mat99::Zero();
// 	for(int j=0;j<9;++j){
// 	    Weight_sqrt(j,j) = imu_weight*sqrt(1/Weight(j,j));
// 	}
// 	LOG(INFO)<<"Weight_sqrt: "<<Weight_sqrt.diagonal().transpose();
	Weight = imu_weight*imu_weight*Weight.inverse();

	Vec9 b_1 = Vec9::Zero();
	b_1.block(0,0,3,1) = res_p2;
	b_1.block(3,0,3,1) = res_phi2;
	b_1.block(6,0,3,1) = res_v2;
	
// 	double resF2_All = 0;
// 	for(EFPoint* p : frames[i]->points){
// 	  for(EFResidual* r : p->residualsAll){
// 	    if(r->isLinearized || !r->isActive()) continue;
// 	    if(r->targetIDX == i+1){
// 	      for(int k=0;k<patternNum;++k){
// 		resF2_All += r->J->resF[k]*r->J->resF[k];
// 	      }
// 	    }
// 	  }
// 	}
// 	for(EFPoint* p : frames[i+1]->points){
// 	  for(EFResidual* r : p->residualsAll){
// 	    if(r->isLinearized || !r->isActive()) continue;
// 	    if(r->targetIDX == i){
// 	      for(int k=0;k<patternNum;++k){
// 		resF2_All += r->J->resF[k]*r->J->resF[k];
// 	      }
// 	    }
// 	  }
// 	}
// 	double E_imu = b_1.transpose()*Weight*b_1;
// 	double bei = sqrt(resF2_All/E_imu);
// 	bei = bei/imu_lambda;
// 	Weight *=(bei*bei);
// 	LOG(INFO)<<"bei"<<bei;
// 	Weight_sqrt = Weight_sqrt*bei/imu_lambda;
// 	LOG(INFO)<<"b_1: "<<b_1.transpose();
// 	LOG(INFO)<<"b_1: "<<b_1.transpose();
	Mat44 T_tempj = T_BC.matrix()*T_WD_l.matrix()*worldToCam_j.matrix();
	Mat1515 J_relj = Mat1515::Identity();
	J_relj.block(0,0,6,6) = (-1*Sim3(T_tempj).Adj()).block(0,0,6,6);
	Mat44 T_tempi = T_BC.matrix()*T_WD_l.matrix()*worldToCam_i.matrix();
	Mat1515 J_reli = Mat1515::Identity();
	J_reli.block(0,0,6,6) = (-1*Sim3(T_tempi).Adj()).block(0,0,6,6);
	
	Mat77 J_poseb_wd_i= Sim3(T_tempi).Adj()-Sim3(T_BC.matrix()*T_WD_l.matrix()).Adj();
	Mat77 J_poseb_wd_j= Sim3(T_tempj).Adj()-Sim3(T_BC.matrix()*T_WD_l.matrix()).Adj();
	J_poseb_wd_i.block(0,0,7,3) = Mat73::Zero();
	J_poseb_wd_j.block(0,0,7,3) = Mat73::Zero();
// 	J_poseb_wd_i.block(0,3,7,3) = Mat73::Zero();
// 	J_poseb_wd_j.block(0,3,7,3) = Mat73::Zero();
// 	J_poseb_wd_i.block(0,6,7,1) = Vec7::Zero();
// 	J_poseb_wd_j.block(0,6,7,1) = Vec7::Zero();
	if(frames.size()<setting_maxFrames){
	    J_poseb_wd_i.block(0,0,7,3) = Mat73::Zero();
	    J_poseb_wd_j.block(0,0,7,3) = Mat73::Zero();
	    J_poseb_wd_i.block(0,3,7,3) = Mat73::Zero();
	    J_poseb_wd_j.block(0,3,7,3) = Mat73::Zero();
	    J_poseb_wd_i.block(0,6,7,1) = Vec7::Zero();
	    J_poseb_wd_j.block(0,6,7,1) = Vec7::Zero();
	}
// 	if(Framei->velocity.norm()>0.1){
// 	    J_poseb_wd_i.block(0,6,7,1) = Vec7::Zero();
// 	    J_poseb_wd_j.block(0,6,7,1) = Vec7::Zero();
// 	}
	
	
// 	LOG(INFO)<<"J_poseb_wd_i: \n"<<J_poseb_wd_i;
// 	LOG(INFO)<<"J_poseb_wd_j: \n"<<J_poseb_wd_j;
	Mat97 J_res_posebi = Mat97::Zero();
	J_res_posebi.block(0,0,9,6) = J_imui.block(0,0,9,6);
	Mat97 J_res_posebj = Mat97::Zero();
	J_res_posebj.block(0,0,9,6) = J_imuj.block(0,0,9,6);
// 	LOG(INFO)<<"5555555";
	Mat66 J_xi_r_l_i = worldToCam_i.Adj().inverse();
	Mat66 J_xi_r_l_j = worldToCam_j.Adj().inverse();
	Mat1515 J_r_l_i = Mat1515::Identity();
	Mat1515 J_r_l_j = Mat1515::Identity();
	J_r_l_i.block(0,0,6,6) = J_xi_r_l_i;
	J_r_l_j.block(0,0,6,6) = J_xi_r_l_j;
	J_all.block(0,0,9,7) += J_res_posebi*J_poseb_wd_i;
	J_all.block(0,0,9,7) += J_res_posebj*J_poseb_wd_j;
	J_all.block(0,0,9,3) = Mat93::Zero();
	
	J_all.block(0,7+i*15,9,15) += J_imui*J_reli*J_r_l_i;
	J_all.block(0,7+(i+1)*15,9,15) += J_imuj*J_relj*J_r_l_j;
	
	r_all.block(0,0,9,1) += b_1;
	
	H += (J_all.transpose()*Weight*J_all);
	b += (J_all.transpose()*Weight*r_all);
	
// 	//bias model
// 	MatXX J_all2 = MatXX::Zero(6, 7+nFrames*15);
// 	VecX r_all2 = VecX::Zero(6);
// 	
// 	r_all2.block(0,0,3,1) = Framej->bias_g+Framej->delta_bias_g - (Framei->bias_g+Framei->delta_bias_g);
// 	r_all2.block(3,0,3,1) = Framej->bias_a+Framej->delta_bias_a - (Framei->bias_a+Framei->delta_bias_a);
// 	
// 	J_all2.block(0,7+i*15+9,3,3) = -Mat33::Identity();
// 	J_all2.block(0,7+(i+1)*15+9,3,3) = Mat33::Identity();
// 	J_all2.block(3,7+i*15+12,3,3) = -Mat33::Identity();
// 	J_all2.block(3,7+(i+1)*15+12,3,3) = Mat33::Identity();
// 	Mat66 Cov_bias = Mat66::Zero();
// 	Cov_bias.block(0,0,3,3) = GyrRandomWalkNoise*dt;
// 	Cov_bias.block(3,3,3,3) = AccRandomWalkNoise*dt;
// 	Mat66 weight_bias = Mat66::Identity()*imu_weight*imu_weight*Cov_bias.inverse();
// // 	weight_bias *= (bei*bei);
// 	H += J_all2.transpose()*weight_bias*J_all2;
// 	b += J_all2.transpose()*weight_bias*r_all2;
// 	LOG(INFO)<<"r_all2: "<<r_all2.transpose();
// 	LOG(INFO)<<"J_all2.transpose()*weight_bias*J_all2: \n"<<J_all2.transpose()*weight_bias*J_all2;
	
	
	Energy = Energy+(r_all.transpose()*Weight*r_all)[0]+(r_all2.transpose()*weight_bias*r_all2)[0];
// 	LOG(INFO)<<"b_1: "<<b_1.transpose();
	
// 	LOG(INFO)<<"Weight_sqrt*b_1: "<<(Weight_sqrt*b_1).transpose();
    }
//     LOG(INFO)<<"IMU Energy: "<<Energy;
//     LOG(INFO)<<"r_all: "<<r_all.transpose();
//     LOG(INFO)<<"666666666";
    
//     LOG(INFO)<<"H: \n"<<H;
//     LOG(INFO)<<"b: \n"<<b.transpose();
//     exit(1);
    
    for(int i=0;i<nFrames;i++){
	H.block(0,7+i*15,7+nFrames*15,3) *= SCALE_XI_TRANS;
	H.block(7+i*15,0,3,7+nFrames*15) *= SCALE_XI_TRANS;
	b.block(7+i*15,0,3,1) *= SCALE_XI_TRANS;
	
	H.block(0,7+i*15+3,7+nFrames*15,3) *= SCALE_XI_ROT;
	H.block(7+i*15+3,0,3,7+nFrames*15) *= SCALE_XI_ROT;
	b.block(7+i*15+3,0,3,1) *= SCALE_XI_ROT;
    }
//     if(nFrames ==3)exit(1);
//     LOG(INFO)<<"H: \n"<<H;
//     LOG(INFO)<<"b: \n"<<b.transpose();
//     if(count_imu_res<3){
// 	H = MatXX::Zero(7+nFrames*15, 7+nFrames*15);
// 	b = VecX::Zero(7+nFrames*15);
//     }
//     LOG(INFO)<<"H_imu: "<<H.diagonal().transpose();
}
void EnergyFunctional::setAdjointsF(CalibHessian* Hcalib)
{

	if(adHost != 0) delete[] adHost;
	if(adTarget != 0) delete[] adTarget;
	adHost = new Mat88[nFrames*nFrames];
	adTarget = new Mat88[nFrames*nFrames];

	for(int h=0;h<nFrames;h++)
		for(int t=0;t<nFrames;t++)
		{
			FrameHessian* host = frames[h]->data;
			FrameHessian* target = frames[t]->data;

			SE3 hostToTarget = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();

			Mat88 AH = Mat88::Identity();
			Mat88 AT = Mat88::Identity();

			AH.topLeftCorner<6,6>() = -hostToTarget.Adj().transpose();
			AT.topLeftCorner<6,6>() = Mat66::Identity();


			Vec2f affLL = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l_0(), target->aff_g2l_0()).cast<float>();
			AT(6,6) = -affLL[0];
			AH(6,6) = affLL[0];
			AT(7,7) = -1;
			AH(7,7) = affLL[0];

			AH.block<3,8>(0,0) *= SCALE_XI_TRANS;
			AH.block<3,8>(3,0) *= SCALE_XI_ROT;
			AH.block<1,8>(6,0) *= SCALE_A;
			AH.block<1,8>(7,0) *= SCALE_B;
			AT.block<3,8>(0,0) *= SCALE_XI_TRANS;
			AT.block<3,8>(3,0) *= SCALE_XI_ROT;
			AT.block<1,8>(6,0) *= SCALE_A;
			AT.block<1,8>(7,0) *= SCALE_B;

			adHost[h+t*nFrames] = AH;
			adTarget[h+t*nFrames] = AT;
		}
	cPrior = VecC::Constant(setting_initialCalibHessian);


	if(adHostF != 0) delete[] adHostF;
	if(adTargetF != 0) delete[] adTargetF;
	adHostF = new Mat88f[nFrames*nFrames];
	adTargetF = new Mat88f[nFrames*nFrames];

	for(int h=0;h<nFrames;h++)
		for(int t=0;t<nFrames;t++)
		{
			adHostF[h+t*nFrames] = adHost[h+t*nFrames].cast<float>();
			adTargetF[h+t*nFrames] = adTarget[h+t*nFrames].cast<float>();
		}

	cPriorF = cPrior.cast<float>();


	EFAdjointsValid = true;
}



EnergyFunctional::EnergyFunctional()
{
	adHost=0;
	adTarget=0;


	red=0;

	adHostF=0;
	adTargetF=0;
	adHTdeltaF=0;

	nFrames = nResiduals = nPoints = 0;

	HM = MatXX::Zero(CPARS,CPARS);
	bM = VecX::Zero(CPARS);
	
	HM_imu = MatXX::Zero(CPARS+7,CPARS+7);
	bM_imu = VecX::Zero(CPARS+7);
	
	HM_bias = MatXX::Zero(CPARS+7,CPARS+7);
	bM_bias = VecX::Zero(CPARS+7);
	
	HM_imu_half = MatXX::Zero(CPARS+7,CPARS+7);
	bM_imu_half = VecX::Zero(CPARS+7);


	accSSE_top_L = new AccumulatedTopHessianSSE();
	accSSE_top_A = new AccumulatedTopHessianSSE();
	accSSE_bot = new AccumulatedSCHessianSSE();

	resInA = resInL = resInM = 0;
	currentLambda=0;
}
EnergyFunctional::~EnergyFunctional()
{
	for(EFFrame* f : frames)
	{
		for(EFPoint* p : f->points)
		{
			for(EFResidual* r : p->residualsAll)
			{
				r->data->efResidual=0;
				delete r;
			}
			p->data->efPoint=0;
			delete p;
		}
		f->data->efFrame=0;
		delete f;
	}

	if(adHost != 0) delete[] adHost;
	if(adTarget != 0) delete[] adTarget;


	if(adHostF != 0) delete[] adHostF;
	if(adTargetF != 0) delete[] adTargetF;
	if(adHTdeltaF != 0) delete[] adHTdeltaF;



	delete accSSE_top_L;
	delete accSSE_top_A;
	delete accSSE_bot;
}




void EnergyFunctional::setDeltaF(CalibHessian* HCalib)
{
	if(adHTdeltaF != 0) delete[] adHTdeltaF;
	adHTdeltaF = new Mat18f[nFrames*nFrames];
	for(int h=0;h<nFrames;h++)
		for(int t=0;t<nFrames;t++)
		{
			int idx = h+t*nFrames;
			adHTdeltaF[idx] = frames[h]->data->get_state_minus_stateZero().head<8>().cast<float>().transpose() * adHostF[idx]
					        +frames[t]->data->get_state_minus_stateZero().head<8>().cast<float>().transpose() * adTargetF[idx];
		}

	cDeltaF = HCalib->value_minus_value_zero.cast<float>();
	for(EFFrame* f : frames)
	{
		f->delta = f->data->get_state_minus_stateZero().head<8>();
		f->delta_prior = (f->data->get_state() - f->data->getPriorZero()).head<8>();

		for(EFPoint* p : f->points)
			p->deltaF = p->data->idepth-p->data->idepth_zero;
	}

	EFDeltaValid = true;
}

// accumulates & shifts L.
void EnergyFunctional::accumulateAF_MT(MatXX &H, VecX &b, bool MT)
{
	if(MT)
	{
		red->reduce(boost::bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_A, nFrames,  _1, _2, _3, _4), 0, 0, 0);
		red->reduce(boost::bind(&AccumulatedTopHessianSSE::addPointsInternal<0>,
				accSSE_top_A, &allPoints, this,  _1, _2, _3, _4), 0, allPoints.size(), 50);
		accSSE_top_A->stitchDoubleMT(red,H,b,this,false,true);
		resInA = accSSE_top_A->nres[0];
	}
	else
	{
		accSSE_top_A->setZero(nFrames);
		for(EFFrame* f : frames)
			for(EFPoint* p : f->points)
				accSSE_top_A->addPoint<0>(p,this);
		accSSE_top_A->stitchDoubleMT(red,H,b,this,false,false);
		resInA = accSSE_top_A->nres[0];
	}
}

// accumulates & shifts L.
void EnergyFunctional::accumulateLF_MT(MatXX &H, VecX &b, bool MT)
{
	if(MT)
	{
		red->reduce(boost::bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_L, nFrames,  _1, _2, _3, _4), 0, 0, 0);
		red->reduce(boost::bind(&AccumulatedTopHessianSSE::addPointsInternal<1>,
				accSSE_top_L, &allPoints, this,  _1, _2, _3, _4), 0, allPoints.size(), 50);
		accSSE_top_L->stitchDoubleMT(red,H,b,this,true,true);
		resInL = accSSE_top_L->nres[0];
	}
	else
	{
		accSSE_top_L->setZero(nFrames);
		for(EFFrame* f : frames)
			for(EFPoint* p : f->points)
				accSSE_top_L->addPoint<1>(p,this);
		accSSE_top_L->stitchDoubleMT(red,H,b,this,true,false);
		resInL = accSSE_top_L->nres[0];
	}
}





void EnergyFunctional::accumulateSCF_MT(MatXX &H, VecX &b, bool MT)
{
	if(MT)
	{
		red->reduce(boost::bind(&AccumulatedSCHessianSSE::setZero, accSSE_bot, nFrames,  _1, _2, _3, _4), 0, 0, 0);
		red->reduce(boost::bind(&AccumulatedSCHessianSSE::addPointsInternal,
				accSSE_bot, &allPoints, true,  _1, _2, _3, _4), 0, allPoints.size(), 50);
		accSSE_bot->stitchDoubleMT(red,H,b,this,true);
	}
	else
	{
		accSSE_bot->setZero(nFrames);
		for(EFFrame* f : frames)
			for(EFPoint* p : f->points)
				accSSE_bot->addPoint(p, true);
		accSSE_bot->stitchDoubleMT(red, H, b,this,false);
	}
}

void EnergyFunctional::resubstituteF_MT(VecX x, CalibHessian* HCalib, bool MT)
{
	assert(x.size() == CPARS+nFrames*8);

	VecXf xF = x.cast<float>();
	HCalib->step = - x.head<CPARS>();

	Mat18f* xAd = new Mat18f[nFrames*nFrames];
	VecCf cstep = xF.head<CPARS>();
	for(EFFrame* h : frames)
	{
		h->data->step.head<8>() = - x.segment<8>(CPARS+8*h->idx);
		h->data->step.tail<2>().setZero();

		for(EFFrame* t : frames)
			xAd[nFrames*h->idx + t->idx] = xF.segment<8>(CPARS+8*h->idx).transpose() *   adHostF[h->idx+nFrames*t->idx]
			            + xF.segment<8>(CPARS+8*t->idx).transpose() * adTargetF[h->idx+nFrames*t->idx];
	}

	if(MT)
		red->reduce(boost::bind(&EnergyFunctional::resubstituteFPt,
						this, cstep, xAd,  _1, _2, _3, _4), 0, allPoints.size(), 50);
	else
		resubstituteFPt(cstep, xAd, 0, allPoints.size(), 0,0);

	delete[] xAd;
}

void EnergyFunctional::resubstituteFPt(
        const VecCf &xc, Mat18f* xAd, int min, int max, Vec10* stats, int tid)
{
	for(int k=min;k<max;k++)
	{
		EFPoint* p = allPoints[k];

		int ngoodres = 0;
// 		for(EFResidual* r : p->residualsAll) if(r->isActive()) ngoodres++;
// 		for(EFResidual* r : p->residualsAll) if(r->isActive()&&r->data->stereoResidualFlag==false) ngoodres++;
		for(EFResidual* r : p->residualsAll) if(r->isActive()) ngoodres++;
		if(ngoodres==0)
		{
			p->data->step = 0;
			continue;
		}
		float b = p->bdSumF;
		b -= xc.dot(p->Hcd_accAF + p->Hcd_accLF);

		for(EFResidual* r : p->residualsAll)
		{
			if(!r->isActive()) continue;
			b -= xAd[r->hostIDX*nFrames + r->targetIDX] * r->JpJdF;
		}

		p->data->step = - b*p->HdiF;
		if(std::isfinite(p->data->step)==false){
		    LOG(INFO)<<"b: "<<b;
		    LOG(INFO)<<"p->HdiF: "<<p->HdiF;
		    LOG(INFO)<<"p->bdSumF: "<<p->bdSumF;
		    LOG(INFO)<<"xc: "<<xc.transpose();
		    LOG(INFO)<<"p->Hcd_accAF: "<<p->Hcd_accAF.transpose()<<" p->Hcd_accLF: "<<p->Hcd_accLF.transpose();
		    LOG(INFO)<<"ngoodres: "<<ngoodres;
		}
		assert(std::isfinite(p->data->step));
	}
}


double EnergyFunctional::calcMEnergyF()
{

	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	VecX delta = getStitchedDeltaF();
	return delta.dot(2*bM + HM*delta);
}


void EnergyFunctional::calcLEnergyPt(int min, int max, Vec10* stats, int tid)
{

	Accumulator11 E;
	E.initialize();
	VecCf dc = cDeltaF;

	for(int i=min;i<max;i++)
	{
		EFPoint* p = allPoints[i];
		float dd = p->deltaF;

		for(EFResidual* r : p->residualsAll)
		{
			if(!r->isLinearized || !r->isActive()) continue;

			Mat18f dp = adHTdeltaF[r->hostIDX+nFrames*r->targetIDX];
			RawResidualJacobian* rJ = r->J;



			// compute Jp*delta
			float Jp_delta_x_1 =  rJ->Jpdxi[0].dot(dp.head<6>())
						   +rJ->Jpdc[0].dot(dc)
						   +rJ->Jpdd[0]*dd;

			float Jp_delta_y_1 =  rJ->Jpdxi[1].dot(dp.head<6>())
						   +rJ->Jpdc[1].dot(dc)
						   +rJ->Jpdd[1]*dd;

			__m128 Jp_delta_x = _mm_set1_ps(Jp_delta_x_1);
			__m128 Jp_delta_y = _mm_set1_ps(Jp_delta_y_1);
			__m128 delta_a = _mm_set1_ps((float)(dp[6]));
			__m128 delta_b = _mm_set1_ps((float)(dp[7]));

			for(int i=0;i+3<patternNum;i+=4)
			{
				// PATTERN: E = (2*res_toZeroF + J*delta) * J*delta.
				__m128 Jdelta =            _mm_mul_ps(_mm_load_ps(((float*)(rJ->JIdx))+i),Jp_delta_x);
				Jdelta = _mm_add_ps(Jdelta,_mm_mul_ps(_mm_load_ps(((float*)(rJ->JIdx+1))+i),Jp_delta_y));
				Jdelta = _mm_add_ps(Jdelta,_mm_mul_ps(_mm_load_ps(((float*)(rJ->JabF))+i),delta_a));
				Jdelta = _mm_add_ps(Jdelta,_mm_mul_ps(_mm_load_ps(((float*)(rJ->JabF+1))+i),delta_b));

				__m128 r0 = _mm_load_ps(((float*)&r->res_toZeroF)+i);
				r0 = _mm_add_ps(r0,r0);
				r0 = _mm_add_ps(r0,Jdelta);
				Jdelta = _mm_mul_ps(Jdelta,r0);
				E.updateSSENoShift(Jdelta);
			}
			for(int i=((patternNum>>2)<<2); i < patternNum; i++)
			{
				float Jdelta = rJ->JIdx[0][i]*Jp_delta_x_1 + rJ->JIdx[1][i]*Jp_delta_y_1 +
								rJ->JabF[0][i]*dp[6] + rJ->JabF[1][i]*dp[7];
				E.updateSingleNoShift((float)(Jdelta * (Jdelta + 2*r->res_toZeroF[i])));
			}
		}
		E.updateSingle(p->deltaF*p->deltaF*p->priorF);
	}
	E.finish();
	(*stats)[0] += E.A;
}




double EnergyFunctional::calcLEnergyF_MT()
{
	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	double E = 0;
	for(EFFrame* f : frames)
        E += f->delta_prior.cwiseProduct(f->prior).dot(f->delta_prior);

	E += cDeltaF.cwiseProduct(cPriorF).dot(cDeltaF);

	red->reduce(boost::bind(&EnergyFunctional::calcLEnergyPt,
			this, _1, _2, _3, _4), 0, allPoints.size(), 50);

	return E+red->stats[0];
}



EFResidual* EnergyFunctional::insertResidual(PointFrameResidual* r)
{
	EFResidual* efr = new EFResidual(r, r->point->efPoint, r->host->efFrame, r->target->efFrame);
	efr->idxInAll = r->point->efPoint->residualsAll.size();
	r->point->efPoint->residualsAll.push_back(efr);
	if(efr->data->stereoResidualFlag == false)
	    connectivityMap[(((uint64_t)efr->host->frameID) << 32) + ((uint64_t)efr->target->frameID)][0]++;
//     connectivityMap[(((uint64_t)efr->host->frameID) << 32) + ((uint64_t)efr->target->frameID)][0]++;
	nResiduals++;
	r->efResidual = efr;
	
	return efr;
}
EFFrame* EnergyFunctional::insertFrame(FrameHessian* fh, CalibHessian* Hcalib)
{
	EFFrame* eff = new EFFrame(fh);
	eff->idx = frames.size();
	frames.push_back(eff);

	nFrames++;
	fh->efFrame = eff;
	//stereo
	EFFrame* eff_right = new EFFrame(fh->frame_right);
	eff_right->idx = frames.size()+10000;
// 	eff_right->idx = frames.size();
	fh->frame_right->efFrame = eff_right;

	assert(HM.cols() == 8*nFrames+CPARS-8);
	bM.conservativeResize(8*nFrames+CPARS);
	HM.conservativeResize(8*nFrames+CPARS,8*nFrames+CPARS);
	bM.tail<8>().setZero();
	HM.rightCols<8>().setZero();
	HM.bottomRows<8>().setZero();
	
	bM_imu.conservativeResize(17*nFrames+CPARS+7);
	HM_imu.conservativeResize(17*nFrames+CPARS+7,17*nFrames+CPARS+7);
	bM_imu.tail<17>().setZero();
	HM_imu.rightCols<17>().setZero();
	HM_imu.bottomRows<17>().setZero();
	
	bM_bias.conservativeResize(17*nFrames+CPARS+7);
	HM_bias.conservativeResize(17*nFrames+CPARS+7,17*nFrames+CPARS+7);
	bM_bias.tail<17>().setZero();
	HM_bias.rightCols<17>().setZero();
	HM_bias.bottomRows<17>().setZero();
	
	bM_imu_half.conservativeResize(17*nFrames+CPARS+7);
	HM_imu_half.conservativeResize(17*nFrames+CPARS+7,17*nFrames+CPARS+7);
	bM_imu_half.tail<17>().setZero();
	HM_imu_half.rightCols<17>().setZero();
	HM_imu_half.bottomRows<17>().setZero();

	EFIndicesValid = false;
	EFAdjointsValid=false;
	EFDeltaValid=false;

	setAdjointsF(Hcalib);
	makeIDX();


	for(EFFrame* fh2 : frames)
	{
        connectivityMap[(((uint64_t)eff->frameID) << 32) + ((uint64_t)fh2->frameID)] = Eigen::Vector2i(0,0);
		if(fh2 != eff)
            connectivityMap[(((uint64_t)fh2->frameID) << 32) + ((uint64_t)eff->frameID)] = Eigen::Vector2i(0,0);
	}

	return eff;
}
EFPoint* EnergyFunctional::insertPoint(PointHessian* ph)
{
	EFPoint* efp = new EFPoint(ph, ph->host->efFrame);
	efp->idxInPoints = ph->host->efFrame->points.size();
	ph->host->efFrame->points.push_back(efp);

	nPoints++;
	ph->efPoint = efp;

	EFIndicesValid = false;

	return efp;
}


void EnergyFunctional::dropResidual(EFResidual* r)
{
	EFPoint* p = r->point;
	assert(r == p->residualsAll[r->idxInAll]);

	p->residualsAll[r->idxInAll] = p->residualsAll.back();
	p->residualsAll[r->idxInAll]->idxInAll = r->idxInAll;
	p->residualsAll.pop_back();


	if(r->isActive())
		r->host->data->shell->statistics_goodResOnThis++;
	else
		r->host->data->shell->statistics_outlierResOnThis++;

	if(r->data->stereoResidualFlag == false)
		connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][0]--;
//     connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][0]--;
	nResiduals--;
	r->data->efResidual=0;
	delete r;
}

void EnergyFunctional::marginalizeFrame_imu(EFFrame* fh){
  
	int ndim = nFrames*17+CPARS+7-17;// new dimension
	int odim = nFrames*17+CPARS+7;// old dimension
	if(nFrames >= setting_maxFrames){
	   imu_track_ready = true;
	}
	
	MatXX HM_change = MatXX::Zero(CPARS+7+nFrames*17, CPARS+7+nFrames*17);
	VecX bM_change = VecX::Zero(CPARS+7+nFrames*17);
	
	MatXX HM_change_half = MatXX::Zero(CPARS+7+nFrames*17, CPARS+7+nFrames*17);
	VecX bM_change_half = VecX::Zero(CPARS+7+nFrames*17);
// 	LOG(INFO)<<"fh->idx: "<<fh->idx;
	double mar_weight = 0.5;
	for(int i=fh->idx-1;i<fh->idx+1;i++){
	    if(i<0)continue;
	    MatXX J_all = MatXX::Zero(9, CPARS+7+nFrames*17);
	    MatXX J_all_half = MatXX::Zero(9, CPARS+7+nFrames*17);
	    VecX r_all = VecX::Zero(9);
	    IMUPreintegrator IMU_preintegrator;
	    IMU_preintegrator.reset();
	    double time_start = pic_time_stamp[frames[i]->data->shell->incoming_id];
	    double time_end = pic_time_stamp[frames[i+1]->data->shell->incoming_id];
	    double dt = time_end-time_start;
	    
	    if(dt>0.5)continue;
	    FrameHessian* Framei = frames[i]->data;
	    FrameHessian* Framej = frames[i+1]->data;
	    
	    SE3 worldToCam_i = Framei->get_worldToCam_evalPT();
	    SE3 worldToCam_j = Framej->get_worldToCam_evalPT();
	    SE3 worldToCam_i2 = Framei->PRE_worldToCam;
	    SE3 worldToCam_j2 = Framej->PRE_worldToCam;

	    int index;
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
		IMU_preintegrator.update(m_gry[index]-Framei->bias_g, m_acc[index]-Framei->bias_a, delta_t);
		if(imu_time_stamp[index+1]>=time_end)
		  break;
		index++;
	    }
	    
	    Vec3 g_w;
	    g_w << 0,0,-G_norm;

// 	    Mat44 M_WB = T_WD.matrix()*worldToCam_i.inverse().matrix()*T_WD.inverse().matrix()*T_BC.inverse().matrix();
// 	    SE3 T_WB(M_WB);
// 	    Mat33 R_WB = T_WB.rotationMatrix();
// 	    Vec3 t_WB = T_WB.translation();
	    
	    Mat44 M_WB2 = T_WD.matrix()*worldToCam_i2.inverse().matrix()*T_WD.inverse().matrix()*T_BC.inverse().matrix();
	    SE3 T_WB2(M_WB2);
	    Mat33 R_WB2 = T_WB2.rotationMatrix();
	    Vec3 t_WB2 = T_WB2.translation();
	    
// 	    Mat44 M_WBj = T_WD.matrix()*worldToCam_j.inverse().matrix()*T_WD.inverse().matrix()*T_BC.inverse().matrix();
// 	    SE3 T_WBj(M_WBj);
// 	    Mat33 R_WBj = T_WBj.rotationMatrix();
// 	    Vec3 t_WBj = T_WBj.translation();
	    
	    Mat44 M_WBj2 = T_WD.matrix()*worldToCam_j2.inverse().matrix()*T_WD.inverse().matrix()*T_BC.inverse().matrix();
	    SE3 T_WBj2(M_WBj2);
	    Mat33 R_WBj2 = T_WBj2.rotationMatrix();
	    Vec3 t_WBj2 = T_WBj2.translation();

	    Mat33 R_temp = SO3::exp(IMU_preintegrator.getJRBiasg()*Framei->delta_bias_g).matrix();
		    
	    Mat33 res_R2 = (IMU_preintegrator.getDeltaR()*R_temp).transpose()*R_WB2.transpose()*R_WBj2;
	    Vec3 res_phi2 = SO3(res_R2).log();
	    Vec3 res_v2 = R_WB2.transpose()*(Framej->velocity-Framei->velocity-g_w*dt)-
		    (IMU_preintegrator.getDeltaV()+IMU_preintegrator.getJVBiasa()*Framei->delta_bias_a+IMU_preintegrator.getJVBiasg()*Framei->delta_bias_g);
	    Vec3 res_p2 = R_WB2.transpose()*(t_WBj2-t_WB2-Framei->velocity*dt-0.5*g_w*dt*dt)-
		    (IMU_preintegrator.getDeltaP()+IMU_preintegrator.getJPBiasa()*Framei->delta_bias_a+IMU_preintegrator.getJPBiasg()*Framei->delta_bias_g);

	    Mat99 Cov = IMU_preintegrator.getCovPVPhi();

// 	    Mat33 J_resPhi_phi_i = -IMU_preintegrator.JacobianRInv(res_phi2)*R_WBj.transpose()*R_WB;
// 	    Mat33 J_resPhi_phi_j = IMU_preintegrator.JacobianRInv(res_phi2);
// 	    Mat33 J_resPhi_bg = -IMU_preintegrator.JacobianRInv(res_phi2)*SO3::exp(-res_phi2).matrix()*
// 		    IMU_preintegrator.JacobianR(IMU_preintegrator.getJRBiasg()*Framei->delta_bias_g)*IMU_preintegrator.getJRBiasg();

// 	    Mat33 J_resV_phi_i = SO3::hat(R_WB.transpose()*(Framej->velocity - Framei->velocity - g_w*dt));
// 	    Mat33 J_resV_v_i = -R_WB.transpose();
// 	    Mat33 J_resV_v_j = R_WB.transpose();
// 	    Mat33 J_resV_ba = -IMU_preintegrator.getJVBiasa();
// 	    Mat33 J_resV_bg = -IMU_preintegrator.getJVBiasg();
// 
// 	    Mat33 J_resP_p_i = -Mat33::Identity();	
// 	    Mat33 J_resP_p_j = R_WB.transpose()*R_WBj;
// 	    Mat33 J_resP_bg = -IMU_preintegrator.getJPBiasg();
// 	    Mat33 J_resP_ba = -IMU_preintegrator.getJPBiasa();
// 	    Mat33 J_resP_v_i = -R_WB.transpose()*dt;
// 	    Mat33 J_resP_phi_i = SO3::hat(R_WB.transpose()*(t_WBj - t_WB - Framei->velocity*dt - 0.5*g_w*dt*dt));
	    
	    Mat33 J_resPhi_phi_i = -IMU_preintegrator.JacobianRInv(res_phi2)*R_WBj2.transpose()*R_WB2;
	    Mat33 J_resPhi_phi_j = IMU_preintegrator.JacobianRInv(res_phi2);
	    Mat33 J_resPhi_bg = -IMU_preintegrator.JacobianRInv(res_phi2)*SO3::exp(-res_phi2).matrix()*
		    IMU_preintegrator.JacobianR(IMU_preintegrator.getJRBiasg()*Framei->delta_bias_g)*IMU_preintegrator.getJRBiasg();

	    Mat33 J_resV_phi_i = SO3::hat(R_WB2.transpose()*(Framej->velocity - Framei->velocity - g_w*dt));
	    Mat33 J_resV_v_i = -R_WB2.transpose();
	    Mat33 J_resV_v_j = R_WB2.transpose();
	    Mat33 J_resV_ba = -IMU_preintegrator.getJVBiasa();
	    Mat33 J_resV_bg = -IMU_preintegrator.getJVBiasg();

	    Mat33 J_resP_p_i = -Mat33::Identity();	
	    Mat33 J_resP_p_j = R_WB2.transpose()*R_WBj2;
	    Mat33 J_resP_bg = -IMU_preintegrator.getJPBiasg();
	    Mat33 J_resP_ba = -IMU_preintegrator.getJPBiasa();
	    Mat33 J_resP_v_i = -R_WB2.transpose()*dt;
	    Mat33 J_resP_phi_i = SO3::hat(R_WB2.transpose()*(t_WBj2 - t_WB2 - Framei->velocity*dt - 0.5*g_w*dt*dt));

	    Mat915 J_imui = Mat915::Zero();//rho,phi,v,bias_g,bias_a;
	    J_imui.block(0,0,3,3) = J_resP_p_i;
	    J_imui.block(0,3,3,3) = J_resP_phi_i;
	    J_imui.block(0,6,3,3) = J_resP_v_i;
	    J_imui.block(0,9,3,3) = J_resP_bg;
	    J_imui.block(0,12,3,3) = J_resP_ba;
	    
	    J_imui.block(3,3,3,3) = J_resPhi_phi_i;
	    J_imui.block(3,9,3,3) = J_resPhi_bg;
	    
	    J_imui.block(6,3,3,3) = J_resV_phi_i;
	    J_imui.block(6,6,3,3) = J_resV_v_i;
	    J_imui.block(6,9,3,3) = J_resV_bg;
	    J_imui.block(6,12,3,3) = J_resV_ba;
	    
	    Mat915 J_imuj = Mat915::Zero();
	    J_imuj.block(0,0,3,3) = J_resP_p_j;
	    J_imuj.block(3,3,3,3) = J_resPhi_phi_j;
	    J_imuj.block(6,6,3,3)  = J_resV_v_j;

	    Mat99 Weight = Mat99::Zero();
	    Weight.block(0,0,3,3) = Cov.block(0,0,3,3);
	    Weight.block(3,3,3,3) = Cov.block(6,6,3,3);
	    Weight.block(6,6,3,3) = Cov.block(3,3,3,3);
	    Mat99 Weight2 = Mat99::Zero();
	    for(int i=0;i<9;++i){
		Weight2(i,i) = Weight(i,i);
	    }
	    Weight = Weight2;
// 	    Mat99 Weight_sqrt = Mat99::Zero();
// 	    for(int j=0;j<9;++j){
// 		Weight_sqrt(j,j) = imu_weight*sqrt(1/Weight(j,j));
// 	    }
	    Weight = imu_weight*imu_weight*Weight.inverse();
    // 	LOG(INFO)<<"Weight_sqrt: "<<Weight_sqrt.diagonal().transpose();
	    Vec9 b_1 = Vec9::Zero();
	    b_1.block(0,0,3,1) = res_p2;
	    b_1.block(3,0,3,1) = res_phi2;
	    b_1.block(6,0,3,1) = res_v2;
	    
// 	    double resF2_All = 0;
// 	    for(EFPoint* p : frames[i]->points){
// 	      for(EFResidual* r : p->residualsAll){
// 		if(r->isLinearized || !r->isActive()) continue;
// 		if(r->targetIDX == i+1){
// 		  for(int k=0;k<patternNum;++k){
// 		    resF2_All += r->J->resF[k]*r->J->resF[k];
// 		  }
// 		}
// 	      }
// 	    }
// 	    for(EFPoint* p : frames[i+1]->points){
// 	      for(EFResidual* r : p->residualsAll){
// 		if(r->isLinearized || !r->isActive()) continue;
// 		if(r->targetIDX == i){
// 		  for(int k=0;k<patternNum;++k){
// 		    resF2_All += r->J->resF[k]*r->J->resF[k];
// 		  }
// 		}
// 	      }
// 	    }
// 	    double E_imu = b_1.transpose()*Weight*b_1;
// 	    double bei = sqrt(resF2_All/E_imu);
// 	    bei = bei/imu_lambda;
// 	    Weight *=(bei*bei);
	   
	    Mat44 T_tempj = T_BC.matrix()*T_WD_l.matrix()*worldToCam_j.matrix();
	    Mat1515 J_relj = Mat1515::Identity();
	    J_relj.block(0,0,6,6) = (-1*Sim3(T_tempj).Adj()).block(0,0,6,6);
	    Mat44 T_tempi = T_BC.matrix()*T_WD_l.matrix()*worldToCam_i.matrix();
	    Mat1515 J_reli = Mat1515::Identity();
	    J_reli.block(0,0,6,6) = (-1*Sim3(T_tempi).Adj()).block(0,0,6,6);
	    
	    Mat44 T_tempj_half = T_BC.matrix()*T_WD_l_half.matrix()*worldToCam_j.matrix();
	    Mat1515 J_relj_half = Mat1515::Identity();
	    J_relj_half.block(0,0,6,6) = (-1*Sim3(T_tempj_half).Adj()).block(0,0,6,6);
	    Mat44 T_tempi_half = T_BC.matrix()*T_WD_l_half.matrix()*worldToCam_i.matrix();
	    Mat1515 J_reli_half = Mat1515::Identity();
	    J_reli_half.block(0,0,6,6) = (-1*Sim3(T_tempi_half).Adj()).block(0,0,6,6);
	    
	    Mat77 J_poseb_wd_i= Sim3(T_tempi).Adj()-Sim3(T_BC.matrix()*T_WD_l.matrix()).Adj();
	    Mat77 J_poseb_wd_j= Sim3(T_tempj).Adj()-Sim3(T_BC.matrix()*T_WD_l.matrix()).Adj();
	    Mat77 J_poseb_wd_i_half= Sim3(T_tempi_half).Adj()-Sim3(T_BC.matrix()*T_WD_l_half.matrix()).Adj();
	    Mat77 J_poseb_wd_j_half= Sim3(T_tempj_half).Adj()-Sim3(T_BC.matrix()*T_WD_l_half.matrix()).Adj();
	    J_poseb_wd_i.block(0,0,7,3) = Mat73::Zero();
	    J_poseb_wd_j.block(0,0,7,3) = Mat73::Zero();
// 	    J_poseb_wd_i.block(0,3,7,3) = Mat73::Zero();
// 	    J_poseb_wd_j.block(0,3,7,3) = Mat73::Zero();
	    J_poseb_wd_i_half.block(0,0,7,3) = Mat73::Zero();
	    J_poseb_wd_j_half.block(0,0,7,3) = Mat73::Zero();
	    
	    Mat97 J_res_posebi = Mat97::Zero();
	    J_res_posebi.block(0,0,9,6) = J_imui.block(0,0,9,6);
	    Mat97 J_res_posebj = Mat97::Zero();
	    J_res_posebj.block(0,0,9,6) = J_imuj.block(0,0,9,6);

	    Mat66 J_xi_r_l_i = worldToCam_i.Adj().inverse();
	    Mat66 J_xi_r_l_j = worldToCam_j.Adj().inverse();
	    Mat1515 J_r_l_i = Mat1515::Identity();
	    Mat1515 J_r_l_j = Mat1515::Identity();
	    J_r_l_i.block(0,0,6,6) = J_xi_r_l_i;
	    J_r_l_j.block(0,0,6,6) = J_xi_r_l_j;
	    
	    J_all.block(0,CPARS,9,7) += J_res_posebi*J_poseb_wd_i;
	    J_all.block(0,CPARS,9,7) += J_res_posebj*J_poseb_wd_j;
	    J_all.block(0,CPARS,9,3) = Mat93::Zero();
	    
	    J_all.block(0,CPARS+7+i*17,9,6) += J_imui.block(0,0,9,6)*J_reli.block(0,0,6,6)*J_xi_r_l_i;
	    J_all.block(0,CPARS+7+(i+1)*17,9,6) += J_imuj.block(0,0,9,6)*J_relj.block(0,0,6,6)*J_xi_r_l_j;
	    J_all.block(0,CPARS+7+i*17+8,9,9) += J_imui.block(0,6,9,9);
	    J_all.block(0,CPARS+7+(i+1)*17+8,9,9) += J_imuj.block(0,6,9,9);
	    
	    J_all_half.block(0,CPARS,9,7) += J_res_posebi*J_poseb_wd_i_half;
	    J_all_half.block(0,CPARS,9,7) += J_res_posebj*J_poseb_wd_j_half;
	    J_all_half.block(0,CPARS,9,3) = Mat93::Zero();
	    
	    J_all_half.block(0,CPARS+7+i*17,9,6) += J_imui.block(0,0,9,6)*J_reli_half.block(0,0,6,6)*J_xi_r_l_i;
	    J_all_half.block(0,CPARS+7+(i+1)*17,9,6) += J_imuj.block(0,0,9,6)*J_relj_half.block(0,0,6,6)*J_xi_r_l_j;
	    J_all_half.block(0,CPARS+7+i*17+8,9,9) += J_imui.block(0,6,9,9);
	    J_all_half.block(0,CPARS+7+(i+1)*17+8,9,9) += J_imuj.block(0,6,9,9);
	    
	    r_all.block(0,0,9,1) += b_1;
	    
	    HM_change += (J_all.transpose()*Weight*J_all);
	    bM_change += (J_all.transpose()*Weight*r_all);
	    
	    HM_change_half = HM_change_half*setting_margWeightFac_imu;
	    bM_change_half = bM_change_half*setting_margWeightFac_imu;
	    
	    MatXX J_all2 = MatXX::Zero(6, CPARS+7+nFrames*17);
	    VecX r_all2 = VecX::Zero(6);
	    r_all2.block(0,0,3,1) = Framej->bias_g+Framej->delta_bias_g - (Framei->bias_g+Framei->delta_bias_g);
	    r_all2.block(3,0,3,1) = Framej->bias_a+Framej->delta_bias_a - (Framei->bias_a+Framei->delta_bias_a);
	    
	    J_all2.block(0,CPARS+7+i*17+8+3,3,3) = -Mat33::Identity();
	    J_all2.block(0,CPARS+7+(i+1)*17+8+3,3,3) = Mat33::Identity();
	    J_all2.block(3,CPARS+7+i*17+8+6,3,3) = -Mat33::Identity();
	    J_all2.block(3,CPARS+7+(i+1)*17+8+6,3,3) = Mat33::Identity();
	    Mat66 Cov_bias = Mat66::Zero();
	    Cov_bias.block(0,0,3,3) = GyrRandomWalkNoise*dt;
	    Cov_bias.block(3,3,3,3) = AccRandomWalkNoise*dt;
	    Mat66 weight_bias = Mat66::Identity()*imu_weight*imu_weight*Cov_bias.inverse();
	    HM_bias += (J_all2.transpose()*weight_bias*J_all2*setting_margWeightFac_imu);
	    bM_bias += (J_all2.transpose()*weight_bias*r_all2*setting_margWeightFac_imu);
	    
// 	    HM_change *= setting_margWeightFac;
// 	    bM_change *= setting_margWeightFac;
	}
	HM_change = HM_change*setting_margWeightFac_imu;
	bM_change = bM_change*setting_margWeightFac_imu;
	
	HM_change_half = HM_change_half*setting_margWeightFac_imu;
	bM_change_half = bM_change_half*setting_margWeightFac_imu;
	
	VecX StitchedDelta = getStitchedDeltaF();
	VecX delta_b = VecX::Zero(CPARS+7+nFrames*17);
	for(int i=fh->idx-1;i<fh->idx+1;i++){
	    if(i<0)continue;
	    double time_start = pic_time_stamp[frames[i]->data->shell->incoming_id];
	    double time_end = pic_time_stamp[frames[i+1]->data->shell->incoming_id];
	    double dt = time_end-time_start;
	    
	    if(dt>0.5)continue;
	    if(i==fh->idx-1){
		delta_b.block(CPARS+7+17*i,0,6,1) = StitchedDelta.block(CPARS+i*8,0,6,1);
		frames[i]->m_flag = true;
	    }
	    if(i==fh->idx){
	      delta_b.block(CPARS+7+17*(i+1),0,6,1) = StitchedDelta.block(CPARS+(i+1)*8,0,6,1);
	      frames[i+1]->m_flag = true;
	    }
	}
// 	for(int i=0;i<nFrames;++i){
// 	    delta_b.block(CPARS+7+17*i,0,6,1) = StitchedDelta.block(CPARS+i*8,0,6,1);
// 	}
	delta_b.block(CPARS,0,7,1) = Sim3(T_WD_l.inverse()*T_WD).log();
	
	    
	VecX delta_b_half = delta_b;
	delta_b_half.block(CPARS,0,7,1) = Sim3(T_WD_l_half.inverse()*T_WD).log();
	
	bM_change -= HM_change*delta_b;
	bM_change_half -= HM_change_half*delta_b_half;
	
	double s_now = T_WD.scale();
	double di=1;
	if(s_last>s_now){
	    di = (s_last+0.001)/s_now;
	}else{
	    di = (s_now+0.001)/s_last;
	}
	s_last = s_now; 
	if(di>d_now)d_now = di;
	if(d_now>d_min) d_now = d_min;
	LOG(INFO)<<"s_now: "<<s_now<<" s_middle: "<<s_middle<<" d_now: "<<d_now<<" scale_l: "<<T_WD_l.scale();
	if(di>d_half)d_half = di;
	if(d_half>d_min) d_half = d_min;
	
	bool side = s_now>s_middle;
	if(side!=side_last||M_num==0){
	    HM_imu_half.block(CPARS+6,0,1,HM_imu_half.cols()) = MatXX::Zero(1,HM_imu_half.cols());
	    HM_imu_half.block(0,CPARS+6,HM_imu_half.rows(),1) = MatXX::Zero(HM_imu_half.rows(),1);
	    bM_imu_half[CPARS+6] = 0;
// 	    HM_imu_half.block(CPARS,0,7,HM_imu_half.cols()) = MatXX::Zero(7,HM_imu_half.cols());
// 	    HM_imu_half.block(0,CPARS,HM_imu_half.rows(),7) = MatXX::Zero(HM_imu_half.rows(),7);
// 	    bM_imu_half.block(CPARS,0,7,1) = VecX::Zero(7);
	    
	    HM_imu_half.setZero();
	    bM_imu_half.setZero();
	    d_half = di;
	    if(d_half>d_min)d_half = d_min;
	    
	    M_num = 0;
	    T_WD_l_half = T_WD;
// 	    LOG(INFO)<<"set half scale: "<<T_WD_l_half.scale();
	}
	M_num++;
	side_last = side;
	
	HM_imu_half += HM_change_half;
 	bM_imu_half += bM_change_half;
	
	if((int)fh->idx != (int)frames.size()-1)
	{
		int io = fh->idx*17+CPARS+7;	// index of frame to move to end
		int ntail = 17*(nFrames-fh->idx-1);
		assert((io+17+ntail) == nFrames*17+CPARS+7);

		Vec17 bTmp = bM_imu_half.segment<17>(io);
		VecX tailTMP = bM_imu_half.tail(ntail);
		bM_imu_half.segment(io,ntail) = tailTMP;
		bM_imu_half.tail<17>() = bTmp;
  
		MatXX HtmpCol = HM_imu_half.block(0,io,odim,17);
		MatXX rightColsTmp = HM_imu_half.rightCols(ntail);
		HM_imu_half.block(0,io,odim,ntail) = rightColsTmp;
		HM_imu_half.rightCols(17) = HtmpCol;

		MatXX HtmpRow = HM_imu_half.block(io,0,17,odim);
		MatXX botRowsTmp = HM_imu_half.bottomRows(ntail);
		HM_imu_half.block(io,0,ntail,odim) = botRowsTmp;
		HM_imu_half.bottomRows(17) = HtmpRow;
	}
	VecX SVec = (HM_imu_half.diagonal().cwiseAbs()+VecX::Constant(HM_imu_half.cols(), 10)).cwiseSqrt();
	VecX SVecI = SVec.cwiseInverse();
	
	MatXX HMScaled = SVecI.asDiagonal() * HM_imu_half * SVecI.asDiagonal();
	VecX bMScaled =  SVecI.asDiagonal() * bM_imu_half;
	
	Mat1717 hpi = HMScaled.bottomRightCorner<17,17>();
	hpi = 0.5f*(hpi+hpi);
	hpi = hpi.inverse();
	hpi = 0.5f*(hpi+hpi);
	if(std::isfinite(hpi(0,0))==false){
	    hpi = Mat1717::Zero();
	}
	
	MatXX bli = HMScaled.bottomLeftCorner(17,ndim).transpose() * hpi;
	HMScaled.topLeftCorner(ndim,ndim).noalias() -= bli * HMScaled.bottomLeftCorner(17,ndim);
	bMScaled.head(ndim).noalias() -= bli*bMScaled.tail<17>();

	//unscale!
	HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
	bMScaled = SVec.asDiagonal() * bMScaled;

	// set.
	HM_imu_half = 0.5*(HMScaled.topLeftCorner(ndim,ndim) + HMScaled.topLeftCorner(ndim,ndim).transpose());
	bM_imu_half = bMScaled.head(ndim);
	
	//marginalize bias
	{
	if((int)fh->idx != (int)frames.size()-1)
	{
		int io = fh->idx*17+CPARS+7;	// index of frame to move to end
		int ntail = 17*(nFrames-fh->idx-1);
		assert((io+17+ntail) == nFrames*17+CPARS+7);

		Vec17 bTmp = bM_bias.segment<17>(io);
		VecX tailTMP = bM_bias.tail(ntail);
		bM_bias.segment(io,ntail) = tailTMP;
		bM_bias.tail<17>() = bTmp;

		MatXX HtmpCol = HM_bias.block(0,io,odim,17);
		MatXX rightColsTmp = HM_bias.rightCols(ntail);
		HM_bias.block(0,io,odim,ntail) = rightColsTmp;
		HM_bias.rightCols(17) = HtmpCol;

		MatXX HtmpRow = HM_bias.block(io,0,17,odim);
		MatXX botRowsTmp = HM_bias.bottomRows(ntail);
		HM_bias.block(io,0,ntail,odim) = botRowsTmp;
		HM_bias.bottomRows(17) = HtmpRow;
	}
	VecX SVec = (HM_bias.diagonal().cwiseAbs()+VecX::Constant(HM_bias.cols(), 10)).cwiseSqrt();
	VecX SVecI = SVec.cwiseInverse();
	
	MatXX HMScaled = SVecI.asDiagonal() * HM_bias * SVecI.asDiagonal();
	VecX bMScaled =  SVecI.asDiagonal() * bM_bias;
	
	Mat1717 hpi = HMScaled.bottomRightCorner<17,17>();
	hpi = 0.5f*(hpi+hpi);
	hpi = hpi.inverse();
	hpi = 0.5f*(hpi+hpi);
	if(std::isfinite(hpi(0,0))==false){
	    hpi = Mat1717::Zero();
	}
	
	MatXX bli = HMScaled.bottomLeftCorner(17,ndim).transpose() * hpi;
	HMScaled.topLeftCorner(ndim,ndim).noalias() -= bli * HMScaled.bottomLeftCorner(17,ndim);
	bMScaled.head(ndim).noalias() -= bli*bMScaled.tail<17>();

	//unscale!
	HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
	bMScaled = SVec.asDiagonal() * bMScaled;

	// set.
	HM_bias = 0.5*(HMScaled.topLeftCorner(ndim,ndim) + HMScaled.topLeftCorner(ndim,ndim).transpose());
	bM_bias = bMScaled.head(ndim);
	}
	if(use_stereo&&M_num2>25){
	    use_Dmargin = false;
	}
	if(use_stereo==false&&M_num2>25){
	    use_Dmargin = false;
	}
// 	if(s_now>s_middle*d_now){
// 	    HM_imu = HM_imu_half;
// 	    bM_imu = bM_imu_half;
// // 	    s_middle = s_middle*d_now;
// 	    s_middle = s_now;
// // 	    d_now = d_half;
// 	    M_num2 = M_num;
// 	    HM_imu_half.setZero();
// 	    bM_imu_half.setZero();
// 	    HM_imu_half.block(CPARS+6,0,1,HM_imu_half.cols()) = MatXX::Zero(1,HM_imu_half.cols());
// // 	    HM_imu_half.block(0,CPARS+6,HM_imu_half.rows(),1) = MatXX::Zero(HM_imu_half.rows(),1);
// // 	    bM_imu_half[CPARS+6] = 0;
// 	    d_half = di;
// 	    if(d_half>d_min)d_half = d_min;
// 	    M_num = 0;
// 	}else if(s_now<s_middle/d_now){
// 	    HM_imu = HM_imu_half;
// 	    bM_imu = bM_imu_half;
// // 	    s_middle = s_middle/d_now;
// 	    s_middle = s_now;
// // 	    d_now = d_half;
// 	    M_num2 = M_num;
// 	    HM_imu_half.setZero();
// 	    bM_imu_half.setZero();
// // 	    HM_imu_half.block(CPARS+6,0,1,HM_imu_half.cols()) = MatXX::Zero(1,HM_imu_half.cols());
// // 	    HM_imu_half.block(0,CPARS+6,HM_imu_half.rows(),1) = MatXX::Zero(HM_imu_half.rows(),1);
// // 	    bM_imu_half[CPARS+6] = 0;
// 	    d_half = di;
// 	    if(d_half>d_min)d_half = d_min;
// 	    M_num = 0;
// 	}else
	  if(/*M_num>20&&*/(s_now>s_middle*d_now||s_now<s_middle/d_now)&&use_Dmargin){
	    HM_imu = HM_imu_half;
	    bM_imu = bM_imu_half;
// 	    s_middle = s_middle/d_now;
	    s_middle = s_now;
// 	    d_now = d_half;
	    M_num2 = M_num;
	    HM_imu_half.setZero();
	    bM_imu_half.setZero();
	    HM_imu_half.block(CPARS+6,0,1,HM_imu_half.cols()) = MatXX::Zero(1,HM_imu_half.cols());
	    HM_imu_half.block(0,CPARS+6,HM_imu_half.rows(),1) = MatXX::Zero(HM_imu_half.rows(),1);
	    bM_imu_half[CPARS+6] = 0;
// 	    HM_imu_half.block(CPARS,0,7,HM_imu_half.cols()) = MatXX::Zero(7,HM_imu_half.cols());
// 	    HM_imu_half.block(0,CPARS,HM_imu_half.rows(),7) = MatXX::Zero(HM_imu_half.rows(),7);
// 	    bM_imu_half.block(CPARS,0,7,1) = VecX::Zero(7);
	    d_half = di;
	    if(d_half>d_min)d_half = d_min;
	    M_num = 0;
	    T_WD_l = T_WD_l_half;
	    state_twd = Sim3(T_WD_l.inverse()*T_WD).log();
	    
// 	    LOG(INFO)<<"set l scale: "<<T_WD_l.scale();
	  }
	  else{
	    HM_imu += HM_change;
	    bM_imu += bM_change;
	    
	    M_num2 ++;
	    if((int)fh->idx != (int)frames.size()-1)
	    {
		    int io = fh->idx*17+CPARS+7;	// index of frame to move to end
		    int ntail = 17*(nFrames-fh->idx-1);
		    assert((io+17+ntail) == nFrames*17+CPARS+7);

		    Vec17 bTmp = bM_imu.segment<17>(io);
		    VecX tailTMP = bM_imu.tail(ntail);
		    bM_imu.segment(io,ntail) = tailTMP;
		    bM_imu.tail<17>() = bTmp;

		    MatXX HtmpCol = HM_imu.block(0,io,odim,17);
		    MatXX rightColsTmp = HM_imu.rightCols(ntail);
		    HM_imu.block(0,io,odim,ntail) = rightColsTmp;
		    HM_imu.rightCols(17) = HtmpCol;

		    MatXX HtmpRow = HM_imu.block(io,0,17,odim);
		    MatXX botRowsTmp = HM_imu.bottomRows(ntail);
		    HM_imu.block(io,0,ntail,odim) = botRowsTmp;
		    HM_imu.bottomRows(17) = HtmpRow;
	    }
	    VecX SVec = (HM_imu.diagonal().cwiseAbs()+VecX::Constant(HM_imu.cols(), 10)).cwiseSqrt();
	    VecX SVecI = SVec.cwiseInverse();
	    
	    MatXX HMScaled = SVecI.asDiagonal() * HM_imu * SVecI.asDiagonal();
	    VecX bMScaled =  SVecI.asDiagonal() * bM_imu;
	    
	    Mat1717 hpi = HMScaled.bottomRightCorner<17,17>();
	    hpi = 0.5f*(hpi+hpi);
	    hpi = hpi.inverse();
	    hpi = 0.5f*(hpi+hpi);
	    if(std::isfinite(hpi(0,0))==false){
		hpi = Mat1717::Zero();
	    }
	    
	    MatXX bli = HMScaled.bottomLeftCorner(17,ndim).transpose() * hpi;
	    HMScaled.topLeftCorner(ndim,ndim).noalias() -= bli * HMScaled.bottomLeftCorner(17,ndim);
	    bMScaled.head(ndim).noalias() -= bli*bMScaled.tail<17>();

	    //unscale!
	    HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
	    bMScaled = SVec.asDiagonal() * bMScaled;

	    // set.
	    HM_imu = 0.5*(HMScaled.topLeftCorner(ndim,ndim) + HMScaled.topLeftCorner(ndim,ndim).transpose());
	    bM_imu = bMScaled.head(ndim);
	}
  
}

void EnergyFunctional::marginalizeFrame(EFFrame* fh)
{

	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	assert((int)fh->points.size()==0);
	
	if(imu_use_flag)
	  marginalizeFrame_imu(fh);
	
	int ndim = nFrames*8+CPARS-8;// new dimension
	int odim = nFrames*8+CPARS;// old dimension


//	VecX eigenvaluesPre = HM.eigenvalues().real();
//	std::sort(eigenvaluesPre.data(), eigenvaluesPre.data()+eigenvaluesPre.size());
//



	if((int)fh->idx != (int)frames.size()-1)
	{
		int io = fh->idx*8+CPARS;	// index of frame to move to end
		int ntail = 8*(nFrames-fh->idx-1);
		assert((io+8+ntail) == nFrames*8+CPARS);

		Vec8 bTmp = bM.segment<8>(io);
		VecX tailTMP = bM.tail(ntail);
		bM.segment(io,ntail) = tailTMP;
		bM.tail<8>() = bTmp;

		MatXX HtmpCol = HM.block(0,io,odim,8);
		MatXX rightColsTmp = HM.rightCols(ntail);
		HM.block(0,io,odim,ntail) = rightColsTmp;
		HM.rightCols(8) = HtmpCol;

		MatXX HtmpRow = HM.block(io,0,8,odim);
		MatXX botRowsTmp = HM.bottomRows(ntail);
		HM.block(io,0,ntail,odim) = botRowsTmp;
		HM.bottomRows(8) = HtmpRow;
	}


//	// marginalize. First add prior here, instead of to active.
    HM.bottomRightCorner<8,8>().diagonal() += fh->prior;
    bM.tail<8>() += fh->prior.cwiseProduct(fh->delta_prior);



//	std::cout << std::setprecision(16) << "HMPre:\n" << HM << "\n\n";


	VecX SVec = (HM.diagonal().cwiseAbs()+VecX::Constant(HM.cols(), 10)).cwiseSqrt();
	VecX SVecI = SVec.cwiseInverse();


//	std::cout << std::setprecision(16) << "SVec: " << SVec.transpose() << "\n\n";
//	std::cout << std::setprecision(16) << "SVecI: " << SVecI.transpose() << "\n\n";

	// scale!
	MatXX HMScaled = SVecI.asDiagonal() * HM * SVecI.asDiagonal();
	VecX bMScaled =  SVecI.asDiagonal() * bM;

	// invert bottom part!
	Mat88 hpi = HMScaled.bottomRightCorner<8,8>();
// 	LOG(INFO)<<"hpi1: \n"<<hpi;
	hpi = 0.5f*(hpi+hpi);
// 	LOG(INFO)<<"hpi2: \n"<<hpi;
	hpi = hpi.inverse();
// 	LOG(INFO)<<"hpi3: \n"<<hpi;
	hpi = 0.5f*(hpi+hpi);
	if(std::isfinite(hpi(0,0))==false){
	    hpi = Mat88::Zero();
	}

	// schur-complement!
	MatXX bli = HMScaled.bottomLeftCorner(8,ndim).transpose() * hpi;
	HMScaled.topLeftCorner(ndim,ndim).noalias() -= bli * HMScaled.bottomLeftCorner(8,ndim);
	bMScaled.head(ndim).noalias() -= bli*bMScaled.tail<8>();

	//unscale!
	HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
	bMScaled = SVec.asDiagonal() * bMScaled;

	// set.
	HM = 0.5*(HMScaled.topLeftCorner(ndim,ndim) + HMScaled.topLeftCorner(ndim,ndim).transpose());
	bM = bMScaled.head(ndim);
// 	if(std::isfinite(HM(0,0))==false){
// 	    LOG(INFO)<<"ndim: "<<ndim;
// 	    LOG(INFO)<<"hpi: \n"<<hpi;
// 	    LOG(INFO)<<"bli: \n"<<bli;
// 	    LOG(INFO)<<"HMScaled: \n"<<HMScaled;
// 	    LOG(INFO)<<"SVecI: \n"<<SVecI.transpose();
// 	    LOG(INFO)<<"SVec: \n"<<SVec.transpose();
// 	    LOG(INFO)<<"fh->prior: \n"<<fh->prior.transpose();
// 	    LOG(INFO)<<"fh->delta_prior: \n"<<fh->delta_prior.transpose();
// 	    exit(1);
// 	}
	// remove from vector, without changing the order!
	for(unsigned int i=fh->idx; i+1<frames.size();i++)
	{
		frames[i] = frames[i+1];
		frames[i]->idx = i;
	}
	frames.pop_back();
	nFrames--;
	fh->data->efFrame=0;

	assert((int)frames.size()*8+CPARS == (int)HM.rows());
	assert((int)frames.size()*8+CPARS == (int)HM.cols());
	assert((int)frames.size()*8+CPARS == (int)bM.size());
	assert((int)frames.size() == (int)nFrames);




//	VecX eigenvaluesPost = HM.eigenvalues().real();
//	std::sort(eigenvaluesPost.data(), eigenvaluesPost.data()+eigenvaluesPost.size());

//	std::cout << std::setprecision(16) << "HMPost:\n" << HM << "\n\n";

//	std::cout << "EigPre:: " << eigenvaluesPre.transpose() << "\n";
//	std::cout << "EigPost: " << eigenvaluesPost.transpose() << "\n";

	EFIndicesValid = false;
	EFAdjointsValid=false;
	EFDeltaValid=false;

	makeIDX();
	delete fh;
}




void EnergyFunctional::marginalizePointsF()
{
	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);


	allPointsToMarg.clear();
	for(EFFrame* f : frames)
	{
		for(int i=0;i<(int)f->points.size();i++)
		{
			EFPoint* p = f->points[i];
			if(p->stateFlag == EFPointStatus::PS_MARGINALIZE)
			{
				p->priorF *= setting_idepthFixPriorMargFac;
				for(EFResidual* r : p->residualsAll)
					if(r->isActive())
						if(r->data->stereoResidualFlag == false)
							 connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][1]++;
//                         connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][1]++;
				int ngoodres = 0;
				for(EFResidual* r : p->residualsAll) if(r->isActive()&&r->data->stereoResidualFlag==false) ngoodres++;
				if(ngoodres>0){
					allPointsToMarg.push_back(p);
				}
				else{
					removePoint(p);
				}
			}
		}
	}
// 	LOG(INFO)<<"allPointsToMarg.size(): "<<allPointsToMarg.size();
	accSSE_bot->setZero(nFrames);
	accSSE_top_A->setZero(nFrames);
	for(EFPoint* p : allPointsToMarg)
	{
		accSSE_top_A->addPoint<2>(p,this);
		accSSE_bot->addPoint(p,false);
		removePoint(p);
	}
	MatXX M, Msc;
	VecX Mb, Mbsc;
	accSSE_top_A->stitchDouble(M,Mb,this,false,false);
	accSSE_bot->stitchDouble(Msc,Mbsc,this);

	resInM+= accSSE_top_A->nres[0];

	MatXX H =  M-Msc;
    VecX b =  Mb-Mbsc;

	if(setting_solverMode & SOLVER_ORTHOGONALIZE_POINTMARG)
	{
		// have a look if prior is there.
		bool haveFirstFrame = false;
		for(EFFrame* f : frames) if(f->frameID==0) haveFirstFrame=true;

		if(!haveFirstFrame)
			orthogonalize(&b, &H);

	}

	HM += setting_margWeightFac*H;
	bM += setting_margWeightFac*b;

	if(setting_solverMode & SOLVER_ORTHOGONALIZE_FULL)
		orthogonalize(&bM, &HM);

	EFIndicesValid = false;
	makeIDX();
}

void EnergyFunctional::dropPointsF()
{


	for(EFFrame* f : frames)
	{
		for(int i=0;i<(int)f->points.size();i++)
		{
			EFPoint* p = f->points[i];
			if(p->stateFlag == EFPointStatus::PS_DROP)
			{
				removePoint(p);
				i--;
			}
		}
	}

	EFIndicesValid = false;
	makeIDX();
}


void EnergyFunctional::removePoint(EFPoint* p)
{
	for(EFResidual* r : p->residualsAll)
		dropResidual(r);

	EFFrame* h = p->host;
	h->points[p->idxInPoints] = h->points.back();
	h->points[p->idxInPoints]->idxInPoints = p->idxInPoints;
	h->points.pop_back();

	nPoints--;
	p->data->efPoint = 0;

	EFIndicesValid = false;

	delete p;
}

void EnergyFunctional::orthogonalize(VecX* b, MatXX* H)
{
//	VecX eigenvaluesPre = H.eigenvalues().real();
//	std::sort(eigenvaluesPre.data(), eigenvaluesPre.data()+eigenvaluesPre.size());
//	std::cout << "EigPre:: " << eigenvaluesPre.transpose() << "\n";


	// decide to which nullspaces to orthogonalize.
	std::vector<VecX> ns;
	ns.insert(ns.end(), lastNullspaces_pose.begin(), lastNullspaces_pose.end());
	ns.insert(ns.end(), lastNullspaces_scale.begin(), lastNullspaces_scale.end());
//	if(setting_affineOptModeA <= 0)
//		ns.insert(ns.end(), lastNullspaces_affA.begin(), lastNullspaces_affA.end());
//	if(setting_affineOptModeB <= 0)
//		ns.insert(ns.end(), lastNullspaces_affB.begin(), lastNullspaces_affB.end());





	// make Nullspaces matrix
	MatXX N(ns[0].rows(), ns.size());
	for(unsigned int i=0;i<ns.size();i++)
		N.col(i) = ns[i].normalized();



	// compute Npi := N * (N' * N)^-1 = pseudo inverse of N.
	Eigen::JacobiSVD<MatXX> svdNN(N, Eigen::ComputeThinU | Eigen::ComputeThinV);

	VecX SNN = svdNN.singularValues();
	double minSv = 1e10, maxSv = 0;
	for(int i=0;i<SNN.size();i++)
	{
		if(SNN[i] < minSv) minSv = SNN[i];
		if(SNN[i] > maxSv) maxSv = SNN[i];
	}
	for(int i=0;i<SNN.size();i++)
		{ if(SNN[i] > setting_solverModeDelta*maxSv) SNN[i] = 1.0 / SNN[i]; else SNN[i] = 0; }

	MatXX Npi = svdNN.matrixU() * SNN.asDiagonal() * svdNN.matrixV().transpose(); 	// [dim] x 9.
	MatXX NNpiT = N*Npi.transpose(); 	// [dim] x [dim].
	MatXX NNpiTS = 0.5*(NNpiT + NNpiT.transpose());	// = N * (N' * N)^-1 * N'.

	if(b!=0) *b -= NNpiTS * *b;
	if(H!=0) *H -= NNpiTS * *H * NNpiTS;


//	std::cout << std::setprecision(16) << "Orth SV: " << SNN.reverse().transpose() << "\n";

//	VecX eigenvaluesPost = H.eigenvalues().real();
//	std::sort(eigenvaluesPost.data(), eigenvaluesPost.data()+eigenvaluesPost.size());
//	std::cout << "EigPost:: " << eigenvaluesPost.transpose() << "\n";

}


void EnergyFunctional::solveSystemF(int iteration, double lambda, CalibHessian* HCalib)
{
	if(setting_solverMode & SOLVER_USE_GN) lambda=0;
	if(setting_solverMode & SOLVER_FIX_LAMBDA) lambda = 1e-5;

	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	MatXX HL_top, HA_top, H_sc;
	VecX  bL_top, bA_top, bM_top, b_sc;

	accumulateAF_MT(HA_top, bA_top,multiThreading);


	accumulateLF_MT(HL_top, bL_top,multiThreading);



	accumulateSCF_MT(H_sc, b_sc,multiThreading);


	bM_top = (bM+ HM * getStitchedDeltaF());
	
	VecX StitchedDelta = getStitchedDeltaF();
	VecX StitchedDelta2 = VecX::Zero(CPARS+7+nFrames*17);
// 	StitchedDelta2.block(0,0,CPARS,1) = StitchedDelta.block(0,0,CPARS,1);
	for(int i=0;i<nFrames;++i){
	    if(frames[i]->m_flag){
		StitchedDelta2.block(CPARS+7+17*i,0,6,1) = StitchedDelta.block(CPARS+8*i,0,6,1);
	    }
	}
// 	for(int i=0;i<nFrames;++i){
// 	    VecX temp = frames[i]->data->get_state();
// 	    StitchedDelta2.block(CPARS+7+17*i,0,6,1) = temp.block(0,0,6,1);
// 	}
	StitchedDelta2.block(CPARS,0,7,1) = state_twd;
	
	VecX bM_top_imu = (bM_imu + HM_imu*StitchedDelta2); 

	MatXX H_imu;
	VecX b_imu;
	getIMUHessian(H_imu,b_imu);



	MatXX HFinal_top;
	VecX bFinal_top;

	if(setting_solverMode & SOLVER_ORTHOGONALIZE_SYSTEM)
	{
		// have a look if prior is there.
		bool haveFirstFrame = false;
		for(EFFrame* f : frames) if(f->frameID==0) haveFirstFrame=true;




		MatXX HT_act =  HL_top + HA_top - H_sc;
		VecX bT_act =   bL_top + bA_top - b_sc;


		if(!haveFirstFrame)
			orthogonalize(&bT_act, &HT_act);

		HFinal_top = HT_act + HM;
		bFinal_top = bT_act + bM_top;





		lastHS = HFinal_top;
		lastbS = bFinal_top;

		for(int i=0;i<8*nFrames+CPARS;i++) HFinal_top(i,i) *= (1+lambda);

	}
	else
	{


		HFinal_top = HL_top + HM + HA_top;
		bFinal_top = bL_top + bM_top + bA_top - b_sc;

		lastHS = HFinal_top - H_sc;
		lastbS = bFinal_top;

		for(int i=0;i<8*nFrames+CPARS;i++) HFinal_top(i,i) *= (1+lambda);
		HFinal_top -= H_sc * (1.0f/(1+lambda));
	}
	
// 	HFinal_top = MatXX::Zero(CPARS+8*nFrames,CPARS+8*nFrames);
// 	bFinal_top = VecX::Zero(CPARS+8*nFrames);
	
	H_imu.block(3,3,3,3) += setting_initialIMUHessian * Mat33::Identity();
	H_imu(6,6) += setting_initialScaleHessian;
	for(int i=0;i<nFrames;++i){
	    H_imu.block(7+15*i+9,7+15*i+9,3,3) += setting_initialbgHessian * Mat33::Identity();
	    H_imu.block(7+15*i+12,7+15*i+12,3,3) += setting_initialbaHessian * Mat33::Identity();
	}
	for(int i=0;i<7+15*nFrames;i++)H_imu(i,i)*= (1+lambda);

		//imu_term
	MatXX HFinal_top2 =  MatXX::Zero(CPARS+7+17*nFrames,CPARS+7+17*nFrames);//Cam,Twd,pose,a,b,v,bg,ba
	VecX bFinal_top2 = VecX::Zero(CPARS+7+17*nFrames);
	HFinal_top2.block(0,0,CPARS,CPARS) = HFinal_top.block(0,0,CPARS,CPARS);
	HFinal_top2.block(CPARS,CPARS,7,7) = H_imu.block(0,0,7,7);
	bFinal_top2.block(0,0,CPARS,1) = bFinal_top.block(0,0,CPARS,1);
	bFinal_top2.block(CPARS,0,7,1) = b_imu.block(0,0,7,1);
	for(int i=0;i<nFrames;++i){
	    //cam
	    HFinal_top2.block(0,CPARS+7+i*17,CPARS,8) += HFinal_top.block(0,CPARS+i*8,CPARS,8);
	    HFinal_top2.block(CPARS+7+i*17,0,8,CPARS) += HFinal_top.block(CPARS+i*8,0,8,CPARS);
	    //Twd
	    HFinal_top2.block(CPARS,CPARS+7+i*17,7,6) += H_imu.block(0,7+i*15,7,6);
	    HFinal_top2.block(CPARS+7+i*17,CPARS,6,7) += H_imu.block(7+i*15,0,6,7);
	    HFinal_top2.block(CPARS,CPARS+7+i*17+8,7,9) += H_imu.block(0,7+i*15+6,7,9);
	    HFinal_top2.block(CPARS+7+i*17+8,CPARS,9,7) += H_imu.block(7+i*15+6,0,9,7);
	    //pose a b
	    HFinal_top2.block(CPARS+7+i*17,CPARS+7+i*17,8,8) += HFinal_top.block(CPARS+i*8,CPARS+i*8,8,8);
	    //pose
	    HFinal_top2.block(CPARS+7+i*17,CPARS+7+i*17,6,6) += H_imu.block(7+i*15,7+i*15,6,6);
	    //v bg ba
	    HFinal_top2.block(CPARS+7+i*17+8,CPARS+7+i*17+8,9,9) += H_imu.block(7+i*15+6,7+i*15+6,9,9);
	    //v bg ba,pose
	    HFinal_top2.block(CPARS+7+i*17+8,CPARS+7+i*17,9,6) += H_imu.block(7+i*15+6,7+i*15,9,6);
	    //pose,v bg ba
	    HFinal_top2.block(CPARS+7+i*17,CPARS+7+i*17+8,6,9) += H_imu.block(7+i*15,7+i*15+6,6,9);
	    
	    for(int j=i+1;j<nFrames;++j){
		//pose a b
		HFinal_top2.block(CPARS+7+i*17,CPARS+7+j*17,8,8) += HFinal_top.block(CPARS+i*8,CPARS+j*8,8,8);
		HFinal_top2.block(CPARS+7+j*17,CPARS+7+i*17,8,8) += HFinal_top.block(CPARS+j*8,CPARS+i*8,8,8);
		//pose
		HFinal_top2.block(CPARS+7+i*17,CPARS+7+j*17,6,6) += H_imu.block(7+i*15,7+j*15,6,6);
		HFinal_top2.block(CPARS+7+j*17,CPARS+7+i*17,6,6) += H_imu.block(7+j*15,7+i*15,6,6);
		//v bg ba
		HFinal_top2.block(CPARS+7+i*17+8,CPARS+7+j*17+8,9,9) += H_imu.block(7+i*15+6,7+j*15+6,9,9);
		HFinal_top2.block(CPARS+7+j*17+8,CPARS+7+i*17+8,9,9) += H_imu.block(7+j*15+6,7+i*15+6,9,9);
		//v bg ba,pose
		HFinal_top2.block(CPARS+7+i*17+8,CPARS+7+j*17,9,6) += H_imu.block(7+i*15+6,7+j*15,9,6);
		HFinal_top2.block(CPARS+7+j*17,CPARS+7+i*17+8,6,9) += H_imu.block(7+j*15,7+i*15+6,6,9);
		//pose,v bg ba
		HFinal_top2.block(CPARS+7+i*17,CPARS+7+j*17+8,6,9) += H_imu.block(7+i*15,7+j*15+6,6,9);
		HFinal_top2.block(CPARS+7+j*17+8,CPARS+7+i*17,9,6) += H_imu.block(7+j*15+6,7+i*15,9,6);		
	    }
	    bFinal_top2.block(CPARS+7+17*i,0,8,1) += bFinal_top.block(CPARS+8*i,0,8,1);
	    bFinal_top2.block(CPARS+7+17*i,0,6,1) += b_imu.block(7+15*i,0,6,1);
	    bFinal_top2.block(CPARS+7+17*i+8,0,9,1) += b_imu.block(7+15*i+6,0,9,1);
	}
	HFinal_top2 += (HM_imu + HM_bias);
// 	bFinal_top2 += (bM_imu + bM_bias);
	bFinal_top2 += (bM_top_imu + bM_bias);
	VecX x = VecX::Zero(CPARS+8*nFrames);
	VecX x2= VecX::Zero(CPARS+7+17*nFrames);
	VecX x3= VecX::Zero(CPARS+7+17*nFrames);
	if(setting_solverMode & SOLVER_SVD)
	{
		VecX SVecI = HFinal_top.diagonal().cwiseSqrt().cwiseInverse();
		MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
		VecX bFinalScaled  = SVecI.asDiagonal() * bFinal_top;
		Eigen::JacobiSVD<MatXX> svd(HFinalScaled, Eigen::ComputeThinU | Eigen::ComputeThinV);

		VecX S = svd.singularValues();
		double minSv = 1e10, maxSv = 0;
		for(int i=0;i<S.size();i++)
		{
			if(S[i] < minSv) minSv = S[i];
			if(S[i] > maxSv) maxSv = S[i];
		}

		VecX Ub = svd.matrixU().transpose()*bFinalScaled;
		int setZero=0;
		for(int i=0;i<Ub.size();i++)
		{
			if(S[i] < setting_solverModeDelta*maxSv)
			{ Ub[i] = 0; setZero++; }

			if((setting_solverMode & SOLVER_SVD_CUT7) && (i >= Ub.size()-7))
			{ Ub[i] = 0; setZero++; }

			else Ub[i] /= S[i];
		}
		x = SVecI.asDiagonal() * svd.matrixV() * Ub;

	}
	else
	{
		if(!imu_use_flag){
		    VecX SVecI = (HFinal_top.diagonal()+VecX::Constant(HFinal_top.cols(), 10)).cwiseSqrt().cwiseInverse();
		    MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
		    x = SVecI.asDiagonal() * HFinalScaled.ldlt().solve(SVecI.asDiagonal() * bFinal_top);//  SVec.asDiagonal() * svd.matrixV() * Ub;
		}
		else{
		    VecX SVecI = (HFinal_top2.diagonal()+VecX::Constant(HFinal_top2.cols(), 10)).cwiseSqrt().cwiseInverse();
		    MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top2 * SVecI.asDiagonal();
		    x2 = SVecI.asDiagonal() * HFinalScaled.ldlt().solve(SVecI.asDiagonal() * bFinal_top2);//  SVec.asDiagonal() * svd.matrixV() * Ub;
// 		    LOG(INFO)<<"HFinal_top2: \n"<<HFinal_top2;
// 		    LOG(INFO)<<"bFinal_top2: "<<bFinal_top2.transpose();
// 		    LOG(INFO)<<"x2: "<<x2.transpose();
// 		    exit(1);
		    x.block(0,0,CPARS,1) = x2.block(0,0,CPARS,1);
		    for(int i=0;i<nFrames;++i){
			x.block(CPARS+i*8,0,8,1) = x2.block(CPARS+7+17*i,0,8,1);
// 			LOG(INFO)<<"x.block(CPARS+i*8,0,8,1): "<<x.block(CPARS+i*8,0,8,1).transpose();
			frames[i]->data->step_imu = -x2.block(CPARS+7+17*i+8,0,9,1);
// 			LOG(INFO)<<"frames[i]->data->step_imu: "<<frames[i]->data->step_imu.transpose();
		    }
		    step_twd = -x2.block(CPARS,0,7,1);
// 		    LOG(INFO)<<"step_twd: "<<step_twd.transpose();
		}
	}



	if((setting_solverMode & SOLVER_ORTHOGONALIZE_X) || (iteration >= 2 && (setting_solverMode & SOLVER_ORTHOGONALIZE_X_LATER)))
	{
		VecX xOld = x;
		orthogonalize(&x, 0);
	}


	lastX = x;
// 	LOG(INFO)<<"x: "<<x.transpose();
// 	exit(1);
// 	if(std::isfinite(x(0))==false){
// 	    LOG(INFO)<<"x: "<<x.transpose();
// 	    LOG(INFO)<<"HA_top: \n"<<HA_top;
// // 	    LOG(INFO)<<"HL_top: \n"<<HL_top;
// 	    LOG(INFO)<<"H_sc: \n"<<H_sc;
// 	    LOG(INFO)<<"HM: \n"<<HM;
// 	    LOG(INFO)<<"bFinal_top: \n"<<bFinal_top.transpose();
// 	}
	//resubstituteF(x, HCalib);
	currentLambda= lambda;
	resubstituteF_MT(x, HCalib,multiThreading);
	currentLambda=0;


}
void EnergyFunctional::makeIDX()
{
	for(unsigned int idx=0;idx<frames.size();idx++)
		frames[idx]->idx = idx;

	allPoints.clear();

	for(EFFrame* f : frames)
		for(EFPoint* p : f->points)
		{
			allPoints.push_back(p);
			for(EFResidual* r : p->residualsAll)
			{
				r->hostIDX = r->host->idx;
				r->targetIDX = r->target->idx;
				if(r->data->stereoResidualFlag == true){
				  r->targetIDX = frames[frames.size()-1]->idx;
				}
			}
		}


	EFIndicesValid=true;
}


VecX EnergyFunctional::getStitchedDeltaF() const
{
	VecX d = VecX(CPARS+nFrames*8); d.head<CPARS>() = cDeltaF.cast<double>();
	for(int h=0;h<nFrames;h++) d.segment<8>(CPARS+8*h) = frames[h]->delta;
	return d;
}



}
