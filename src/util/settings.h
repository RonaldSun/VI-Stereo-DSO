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



#pragma once

#include <string.h>
#include <string>
#include <cmath>
#include <vector>
#include "NumType.h"


namespace dso
{
#define SOLVER_SVD (int)1
#define SOLVER_ORTHOGONALIZE_SYSTEM (int)2
#define SOLVER_ORTHOGONALIZE_POINTMARG (int)4
#define SOLVER_ORTHOGONALIZE_FULL (int)8
#define SOLVER_SVD_CUT7 (int)16
#define SOLVER_REMOVE_POSEPRIOR (int)32
#define SOLVER_USE_GN (int)64
#define SOLVER_FIX_LAMBDA (int)128
#define SOLVER_ORTHOGONALIZE_X (int)256
#define SOLVER_MOMENTUM (int)512
#define SOLVER_STEPMOMENTUM (int)1024
#define SOLVER_ORTHOGONALIZE_X_LATER (int)2048


// ============== PARAMETERS TO BE DECIDED ON COMPILE TIME =================
#define PYR_LEVELS 6
extern int pyrLevelsUsed;



extern float setting_keyframesPerSecond;
extern bool setting_realTimeMaxKF;
extern float setting_maxShiftWeightT;
extern float setting_maxShiftWeightR;
extern float setting_maxShiftWeightRT;
extern float setting_maxAffineWeight;
extern float setting_kfGlobalWeight;



extern float setting_idepthFixPrior;
extern float setting_idepthFixPriorMargFac;
extern float setting_initialRotPrior;
extern float setting_initialTransPrior;
extern float setting_initialAffBPrior;
extern float setting_initialAffAPrior;
extern float setting_initialCalibHessian;
extern float setting_initialIMUHessian;
extern float setting_initialScaleHessian;
extern float setting_initialbaHessian;
extern float setting_initialbgHessian;

extern int setting_solverMode;
extern double setting_solverModeDelta;


extern float setting_minIdepthH_act;
extern float setting_minIdepthH_marg;



extern float setting_maxIdepth;
extern float setting_maxPixSearch;
extern float setting_desiredImmatureDensity;			// done
extern float setting_desiredPointDensity;			// done
extern float setting_minPointsRemaining;
extern float setting_maxLogAffFacInWindow;
extern int setting_minFrames;
extern int setting_maxFrames;
extern int setting_minFrameAge;
extern int setting_maxOptIterations;
extern int setting_minOptIterations;
extern float setting_thOptIterations;
extern float setting_outlierTH;
extern float setting_outlierTHSumComponent;



extern int setting_pattern;
extern float setting_margWeightFac;
extern int setting_GNItsOnPointActivation;


extern float setting_minTraceQuality;
extern int setting_minTraceTestRadius;
extern float setting_reTrackThreshold;


extern int   setting_minGoodActiveResForMarg;
extern int   setting_minGoodResForMarg;
extern int   setting_minInlierVotesForMarg;




extern int setting_photometricCalibration;
extern bool setting_useExposure;
extern float setting_affineOptModeA;
extern float setting_affineOptModeB;
extern int setting_gammaWeightsPixelSelect;



extern bool setting_forceAceptStep;



extern float setting_huberTH;


extern bool setting_logStuff;
extern float benchmarkSetting_fxfyfac;
extern int benchmarkSetting_width;
extern int benchmarkSetting_height;
extern float benchmark_varNoise;
extern float benchmark_varBlurNoise;
extern int benchmark_noiseGridsize;
extern float benchmark_initializerSlackFactor;

extern float setting_frameEnergyTHConstWeight;
extern float setting_frameEnergyTHN;

extern float setting_frameEnergyTHFacMedian;
extern float setting_overallEnergyTHWeight;
extern float setting_coarseCutoffTH;

extern float setting_minGradHistCut;
extern float setting_minGradHistAdd;
extern float setting_gradDownweightPerLevel;
extern bool  setting_selectDirectionDistribution;



extern float setting_trace_stepsize;
extern int setting_trace_GNIterations;
extern float setting_trace_GNThreshold;
extern float setting_trace_extraSlackOnTH;
extern float setting_trace_slackInterval;
extern float setting_trace_minImprovementFactor;


extern bool setting_render_displayCoarseTrackingFull;
extern bool setting_render_renderWindowFrames;
extern bool setting_render_plotTrackingFull;
extern bool setting_render_display3D;
extern bool setting_render_displayResidual;
extern bool setting_render_displayVideo;
extern bool setting_render_displayDepth;

extern bool setting_fullResetRequested;

extern bool setting_debugout_runquiet;

extern bool disableAllDisplay;
extern bool disableReconfigure;


extern bool setting_onlyLogKFPoses;




extern bool debugSaveImages;


extern int sparsityFactor;
extern bool goStepByStep;
extern bool plotStereoImages;
extern bool multiThreading;

extern float freeDebugParam1;
extern float freeDebugParam2;
extern float freeDebugParam3;
extern float freeDebugParam4;
extern float freeDebugParam5;


void handleKey(char k);




extern int staticPattern[10][40][2];
extern int staticPatternNum[10];
extern int staticPatternPadding[10];

extern double baseline;
extern std::string gt_path;
extern std::string imu_path;
extern std::vector<SE3> gt_pose;
extern std::vector<Vec3> gt_velocity;
extern std::vector<Vec3> gt_bias_g;
extern std::vector<Vec3> gt_bias_a;
extern std::vector<Vec3> m_gry;
extern std::vector<Vec3> m_acc;
extern std::string savefile_tail;
extern SE3 T_C0C1;
extern SE3 T_C1C0;
extern Mat33f K_right;
extern std::vector<double> gt_time_stamp;
extern std::vector<double> imu_time_stamp;
extern std::vector<double> pic_time_stamp;
extern std::vector<double> pic_time_stamp_r;
extern SE3 T_BC;
extern Mat33 GyrCov;
extern Mat33 AccCov;
extern Mat33 GyrRandomWalkNoise;
extern Mat33 AccRandomWalkNoise;
extern Sim3 T_WD;
extern Sim3 T_WD_l;
extern Sim3 T_WD_l_half;
extern Sim3 T_WD_change;
extern double G_norm;
extern int index_align;
extern SE3 T_WR_align;
extern double run_time;
extern Vec7 step_twd;
extern Vec7 state_twd;
extern double imu_weight;
extern double imu_weight_tracker;
extern bool imu_use_flag;
extern bool imu_track_flag;
extern bool use_optimize;
extern bool use_Dmargin;
extern double d_min;
extern double imu_lambda;
extern bool imu_track_ready;
extern double stereo_weight;
extern int M_num;
extern int M_num2;
extern double setting_margWeightFac_imu;
extern bool first_track_flag;
extern bool use_stereo;


//#define patternNum staticPatternNum[setting_pattern]
//#define patternP staticPattern[setting_pattern]
//#define patternPadding staticPatternPadding[setting_pattern]

//
#define patternNum 8
#define patternP staticPattern[8]
#define patternPadding 2













}
