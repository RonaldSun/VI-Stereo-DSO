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



#include <thread>
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "IOWrapper/Output3DWrapper.h"
#include "IOWrapper/ImageDisplay.h"


#include <boost/thread.hpp>
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/DatasetReader.h"
#include "util/globalCalib.h"

#include "util/NumType.h"
#include "FullSystem/FullSystem.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "FullSystem/PixelSelector2.h"



#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"
#include <opencv2/highgui/highgui.hpp>


std::string vignette = "";
std::string gammaCalib = "";
std::string source = "";
std::string source0 = "";
std::string source1 = "";
std::string calib = "";
std::string calib0 = "";
std::string calib1 = "";
std::string T_stereo = "";
std::string imu_info = "";
std::string pic_timestamp = "";
std::string pic_timestamp1 = "";
double rescale = 1;
bool reverse = false;
bool disableROS = false;
int start=0;
int end=100000;
bool prefetch = false;
float playbackSpeed=0;	// 0 for linearize (play as fast as possible, while sequentializing tracking & mapping). otherwise, factor on timestamps.
bool preload=false;
bool useSampleOutput=false;


int mode=0;

bool firstRosSpin=false;

using namespace dso;


void my_exit_handler(int s)
{
	printf("Caught signal %d\n",s);
	exit(1);
}

void exitThread()
{
	struct sigaction sigIntHandler;
	sigIntHandler.sa_handler = my_exit_handler;
	sigemptyset(&sigIntHandler.sa_mask);
	sigIntHandler.sa_flags = 0;
	sigaction(SIGINT, &sigIntHandler, NULL);

	firstRosSpin=true;
	while(true) pause();
}



void settingsDefault(int preset)
{
	printf("\n=============== PRESET Settings: ===============\n");
	if(preset == 0 || preset == 1)
	{
		printf("DEFAULT settings:\n"
				"- %s real-time enforcing\n"
				"- 2000 active points\n"
				"- 5-7 active frames\n"
				"- 1-6 LM iteration each KF\n"
				"- original image resolution\n", preset==0 ? "no " : "1x");

		playbackSpeed = (preset==0 ? 0 : 1);
		preload = preset==1;
		setting_desiredImmatureDensity = 1500;
		setting_desiredPointDensity = 2000;
// 		setting_desiredImmatureDensity = 3000;
// 		setting_desiredPointDensity = 3000;
		setting_minFrames = 5;
		setting_maxFrames = 7;
		setting_maxOptIterations=6;
		setting_minOptIterations=1;

		setting_logStuff = false;
		setting_kfGlobalWeight=1;   // original is 1.0. 0.3 is a balance between speed and accuracy. if tracking lost, set this para higher
		setting_maxShiftWeightT= 0.04f * (640 + 128);   // original is 0.04f * (640+480); this para is depend on the crop size.
		setting_maxShiftWeightR= 0.04f * (640 + 128);    // original is 0.0f * (640+480);
		setting_maxShiftWeightRT= 0.02f * (640 + 128);  // original is 0.02f * (640+480);
	}

	if(preset == 2 || preset == 3)
	{
		printf("FAST settings:\n"
				"- %s real-time enforcing\n"
				"- 800 active points\n"
				"- 4-6 active frames\n"
				"- 1-4 LM iteration each KF\n"
				"- 424 x 320 image resolution\n", preset==0 ? "no " : "5x");

		playbackSpeed = (preset==2 ? 0 : 5);
		preload = preset==3;
		setting_desiredImmatureDensity = 600;
		setting_desiredPointDensity = 800;
		setting_minFrames = 4;
		setting_maxFrames = 6;
		setting_maxOptIterations=4;
		setting_minOptIterations=1;

		benchmarkSetting_width = 424;
		benchmarkSetting_height = 320;

		setting_logStuff = false;
	}

	printf("==============================================\n");
}






void parseArgument(char* arg)
{
	int option;
	float foption;
	char buf[1000];


    if(1==sscanf(arg,"sampleoutput=%d",&option))
    {
        if(option==1)
        {
            useSampleOutput = true;
            printf("USING SAMPLE OUTPUT WRAPPER!\n");
        }
        return;
    }

    if(1==sscanf(arg,"quiet=%d",&option))
    {
        if(option==1)
        {
            setting_debugout_runquiet = true;
            printf("QUIET MODE, I'll shut up!\n");
        }
        return;
    }

	if(1==sscanf(arg,"preset=%d",&option))
	{
		settingsDefault(option);
		return;
	}


	if(1==sscanf(arg,"rec=%d",&option))
	{
		if(option==0)
		{
			disableReconfigure = true;
			printf("DISABLE RECONFIGURE!\n");
		}
		return;
	}



	if(1==sscanf(arg,"noros=%d",&option))
	{
		if(option==1)
		{
			disableROS = true;
			disableReconfigure = true;
			printf("DISABLE ROS (AND RECONFIGURE)!\n");
		}
		return;
	}

	if(1==sscanf(arg,"nolog=%d",&option))
	{
		if(option==1)
		{
			setting_logStuff = false;
			printf("DISABLE LOGGING!\n");
		}
		return;
	}
	if(1==sscanf(arg,"reverse=%d",&option))
	{
		if(option==1)
		{
			reverse = true;
			printf("REVERSE!\n");
		}
		return;
	}
	if(1==sscanf(arg,"nogui=%d",&option))
	{
		if(option==1)
		{
			disableAllDisplay = true;
			printf("NO GUI!\n");
		}
		return;
	}
	if(1==sscanf(arg,"nomt=%d",&option))
	{
		if(option==1)
		{
			multiThreading = false;
			printf("NO MultiThreading!\n");
		}
		return;
	}
	if(1==sscanf(arg,"prefetch=%d",&option))
	{
		if(option==1)
		{
			prefetch = true;
			printf("PREFETCH!\n");
		}
		return;
	}
	if(1==sscanf(arg,"start=%d",&option))
	{
		start = option;
		printf("START AT %d!\n",start);
		return;
	}
	if(1==sscanf(arg,"end=%d",&option))
	{
		end = option;
		printf("END AT %d!\n",start);
		return;
	}

	if(1==sscanf(arg,"files=%s",buf))
	{
		source = buf;
		printf("loading data from %s!\n", source.c_str());
		return;
	}
	if(1==sscanf(arg,"files0=%s",buf))
	{
		source0 = buf;
		printf("loading data from %s!\n", source0.c_str());
		return;
	}
	if(1==sscanf(arg,"files1=%s",buf))
	{
		source1 = buf;
		printf("loading data from %s!\n", source1.c_str());
		return;
	}
	if(1==sscanf(arg,"groundtruth=%s",buf))
	{
		gt_path = buf;
		printf("loading groundtruth from %s!\n", gt_path.c_str());
		return;
	}
	if(1==sscanf(arg,"imudata=%s",buf))
	{
		imu_path = buf;
		printf("loading groundtruth from %s!\n", imu_path.c_str());
		return;
	}
	if(1==sscanf(arg,"savefile_tail=%s",buf))
	{
		savefile_tail = buf;
		return;
	}

	if(1==sscanf(arg,"calib=%s",buf))
	{
		calib = buf;
		printf("loading calibration from %s!\n", calib.c_str());
		return;
	}
	
	if(1==sscanf(arg,"calib0=%s",buf))
	{
		calib0 = buf;
		printf("loading calibration from %s!\n", calib0.c_str());
		return;
	}
	
	if(1==sscanf(arg,"T_stereo=%s",buf))
	{
		T_stereo = buf;
		return;
	}
	
	if(1==sscanf(arg,"imu_info=%s",buf))
	{
		imu_info = buf;
		return;
	}
	
	if(1==sscanf(arg,"pic_timestamp=%s",buf))
	{
		pic_timestamp = buf;
		return;
	}
	
	if(1==sscanf(arg,"pic_timestamp1=%s",buf))
	{
		pic_timestamp1 = buf;
		return;
	}
	
	if(1==sscanf(arg,"calib1=%s",buf))
	{
		calib1 = buf;
		printf("loading calibration from %s!\n", calib1.c_str());
		return;
	}

	if(1==sscanf(arg,"vignette=%s",buf))
	{
		vignette = buf;
		printf("loading vignette from %s!\n", vignette.c_str());
		return;
	}

	if(1==sscanf(arg,"gamma=%s",buf))
	{
		gammaCalib = buf;
		printf("loading gammaCalib from %s!\n", gammaCalib.c_str());
		return;
	}

	if(1==sscanf(arg,"rescale=%f",&foption))
	{
		rescale = foption;
		printf("RESCALE %f!\n", rescale);
		return;
	}
	
	if(1==sscanf(arg,"imu_weight=%f",&foption))
	{
		imu_weight = foption;
		return;
	}
	if(1==sscanf(arg,"stereo_weight=%f",&foption))
	{
		stereo_weight = foption;
		return;
	}
	if(1==sscanf(arg,"imu_weight_tracker=%f",&foption))
	{
		imu_weight_tracker = foption;
		return;
	}
	
	if(1==sscanf(arg,"use_stereo=%d",&option))
	{
		if(option==0)
		{
			use_stereo = false;
			printf("NO GUI!\n");
		}
		else{
			use_stereo = true;
		}
		return;
	}

	if(1==sscanf(arg,"speed=%f",&foption))
	{
		playbackSpeed = foption;
		printf("PLAYBACK SPEED %f!\n", playbackSpeed);
		return;
	}

	if(1==sscanf(arg,"save=%d",&option))
	{
		if(option==1)
		{
			debugSaveImages = true;
			if(42==system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if(42==system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if(42==system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if(42==system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			printf("SAVE IMAGES!\n");
		}
		return;
	}

	if(1==sscanf(arg,"mode=%d",&option))
	{

		mode = option;
		if(option==0)
		{
			printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
		}
		if(option==1)
		{
			printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
			setting_photometricCalibration = 0;
			setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
			setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).
		}
		if(option==2)
		{
			printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
			setting_photometricCalibration = 0;
			setting_affineOptModeA = -1; //-1: fix. >=0: optimize (with prior, if > 0).
			setting_affineOptModeB = -1; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_minGradHistAdd=3;
		}
		return;
	}

	printf("could not parse argument \"%s\"!!!!\n", arg);
}

void getGroundtruth_kitti(){
	std::ifstream inf;
	inf.open(gt_path);
	std::string sline;
	std::getline(inf,sline);
	while(std::getline(inf,sline)){
		std::istringstream ss(sline);
		Mat33 R;
		Vec3 t;
		for(int i=0;i<3;++i){
			for(int j=0;j<3;++j){
				ss>>R(i,j);			      
			}
			ss>>t(i);
		}
		SE3 temp(R,t);
		gt_pose.push_back(temp);
	}
	inf.close();
}

Eigen::Matrix3d quaternionToRotation(const Eigen::Vector4d& q) {
    Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
    
    R(0, 0) = 1-2.0*q(1)*q(1)-2.0*q(2)*q(2);
    R(0, 1) = 2.0*(q(0)*q(1) - q(2)*q(3));
    R(0, 2) = 2.0*(q(0)*q(2) + q(1)*q(3));
    
    R(1, 0) = 2.0*(q(0)*q(1) + q(2)*q(3));
    R(1, 1) = -1*q(0)*q(0) + q(1)*q(1) - q(2)*q(2) + q(3)*q(3);
    R(1, 2) = 2.0*(q(1)*q(2) - q(0)*q(3));
    
    R(2, 0) = 2.0*(q(0)*q(2) - q(1)*q(3));
    R(2, 1) = 2.0*(q(1)*q(2) + q(0)*q(3));
    R(2, 2) = -1*q(0)*q(0) - q(1)*q(1) + q(2)*q(2) + q(3)*q(3);
    return R;
}

void getGroundtruth_euroc(){
	std::ifstream inf;
	
	if(gt_path.size() == 0)
	    return;
	inf.open(gt_path);
	std::string sline;
	std::getline(inf,sline);
	while(std::getline(inf,sline)){
		std::istringstream ss(sline);
		Vec4 q4;
		Vec3 t;
		Vec3 v;
		Vec3 bias_g;
		Vec3 bias_a;
		double time;
		ss>>time;
		time = time/1e9;
		char temp;
		for(int i=0;i<3;++i){
		  ss>>temp;
		  ss>>t(i);
		}
		ss>>temp;
		ss>>q4(3);
		for(int i=0;i<3;++i){
		  ss>>temp;
		  ss>>q4(i);
		}
		for(int i=0;i<3;++i){
		  ss>>temp;
		  ss>>v(i);
		}
		for(int i=0;i<3;++i){
		  ss>>temp;
		  ss>>bias_g(i);
		}
		for(int i=0;i<3;++i){
		  ss>>temp;
		  ss>>bias_a(i);
		}
		Eigen::Matrix3d R_wb = quaternionToRotation(q4);
		SE3 pose0(R_wb,t);
		gt_pose.push_back(pose0);
		gt_time_stamp.push_back(time);
		gt_velocity.push_back(v);
		gt_bias_g.push_back(bias_g);
		gt_bias_a.push_back(bias_a);
	}
	inf.close();
}
void getIMUdata_euroc(){
	std::ifstream inf;
	inf.open(imu_path);
	std::string sline;
	std::getline(inf,sline);
	while(std::getline(inf,sline)){
		std::istringstream ss(sline);
		Vec3 gyro,acc;
		double time;
		ss>>time;
		time = time/1e9;
		char temp;
		for(int i=0;i<3;++i){
		  ss>>temp;
		  ss>>gyro(i);
		}
		for(int i=0;i<3;++i){
		  ss>>temp;
		  ss>>acc(i);
		}
		m_gry.push_back(gyro);
		m_acc.push_back(acc);
		imu_time_stamp.push_back(time);
	}
	inf.close();
}

void getTstereo(){
	std::ifstream inf;
	inf.open(T_stereo);
	std::string sline;
	int line = 0;
	Mat33 R;
	Vec3 t;
	while(line<3&&std::getline(inf,sline)){
		std::istringstream ss(sline);
		for(int i=0;i<3;++i){
		    ss>>R(line,i);
		}
		ss>>t(line);
		++line;
	}
	inf.close();
	SE3 temp(R,t);
	T_C0C1 = temp;
	T_C1C0 = temp.inverse();
}
void getIMUinfo(){
	std::ifstream inf;
	inf.open(imu_info);
	std::string sline;
	int line = 0;
	Mat33 R;
	Vec3 t;
	Vec4 noise;
	while(line<3&&std::getline(inf,sline)){
		std::istringstream ss(sline);
		for(int i=0;i<3;++i){
		    ss>>R(line,i);
		}
		ss>>t(line);
		++line;
	}
	std::getline(inf,sline);
	++line;
	while(line<8&&std::getline(inf,sline)){
		std::istringstream ss(sline);
		ss>>noise(line-4);
		++line;
	}
	SE3 temp(R,t);
	T_BC = temp;
	
	GyrCov = Mat33::Identity()*noise(0)*noise(0)/0.005;
	AccCov = Mat33::Identity()*noise(1)*noise(1)/0.005;
	GyrRandomWalkNoise = Mat33::Identity()*noise(2)*noise(2);
	AccRandomWalkNoise = Mat33::Identity()*noise(3)*noise(3);
	
	LOG(INFO)<<"T_BC: \n"<<T_BC.matrix();
	LOG(INFO)<<"noise: "<<noise.transpose();
	inf.close();
	
}

void getPicTimestamp(){
	std::ifstream inf;
	inf.open(pic_timestamp);
	std::string sline;
	std::getline(inf,sline);
	while(std::getline(inf,sline)){
		std::istringstream ss(sline);
		double time;
		ss>>time;
		time = time/1e9;
		pic_time_stamp.push_back(time);
	}
	inf.close();
	if(pic_timestamp1.size()>0){
	    std::ifstream inf;
	    inf.open(pic_timestamp1);
	    std::string sline;
	    std::getline(inf,sline);
	    while(std::getline(inf,sline)){
		    std::istringstream ss(sline);
		    double time;
		    ss>>time;
		    time = time/1e9;
		    pic_time_stamp_r.push_back(time);
	    }
	    inf.close();
	}
}

int main( int argc, char** argv )
{
	//setlocale(LC_ALL, "");
	imu_weight = 3;
	imu_weight_tracker = 0.1;
	stereo_weight = 2;
	for(int i=1; i<argc;i++)
		parseArgument(argv[i]);

	if(gt_path.size()>0)getGroundtruth_euroc();
	if(T_stereo.size()>0)getTstereo();
	if(imu_info.size()>0)getIMUinfo();
	
// 	Mat33 R_BC;
// 	R_BC<<0.0148655429818,-0.999880929698,0.00414029679422,0.999557249008,0.0149672133247,0.025715529948,-0.0257744366974,0.00375618835797,0.999660727178;
// 	Vec3 t_BC;
// 	t_BC<<-0.0216401454975,-0.064676986768,0.00981073058949;
// 	T_BC = SE3(R_BC,t_BC);
	
// 	GyrCov = Mat33::Identity()*1.6968e-04*1.6968e-04/0.005;
// 	AccCov = Mat33::Identity()*2.0000e-3*2.0000e-3/0.005;
// 	GyrRandomWalkNoise = Mat33::Identity()*1.9393e-05*1.9393e-05;
// 	AccRandomWalkNoise = Mat33::Identity()*3.0000e-3*3.0000e-3;
	
	G_norm = 9.81;
	imu_use_flag = true;
	imu_track_flag = true;
	use_optimize = true;
	imu_track_ready = false;
	use_Dmargin = true;
	setting_initialIMUHessian = 0;
	setting_initialScaleHessian = 0;
	setting_initialbaHessian = 0;
	setting_initialbgHessian = 0;
	imu_lambda = 5;
	d_min = sqrt(1.1);
	setting_margWeightFac_imu = 0.25;
	
	getIMUdata_euroc();
	getPicTimestamp();
	
	double time_start;
	
	
	// hook crtl+C.
	boost::thread exThread = boost::thread(exitThread);


	ImageFolderReader* reader = new ImageFolderReader(source0, calib0, gammaCalib, vignette);
	ImageFolderReader* reader_right;
	if(use_stereo)
	  reader_right= new ImageFolderReader(source1, calib1, gammaCalib, vignette);
	else
	  reader_right= new ImageFolderReader(source0, calib0, gammaCalib, vignette);
	reader->setGlobalCalibration();
// 	reader_right->setGlobalCalibration();
	int w_out, h_out;
	reader_right->getCalibMono(K_right,w_out,h_out);
	
	LOG(INFO)<<"K_right: \n"<<K_right;
// 	LOG(INFO)<<"T_C0C1: \n"<<T_C0C1.matrix();
// 	exit(1);

	if(setting_photometricCalibration > 0 && reader->getPhotometricGamma() == 0)
	{
		printf("ERROR: dont't have photometric calibation. Need to use commandline options mode=1 or mode=2 ");
		exit(1);
	}



	int lstart=start;
	int lend = end;
	int linc = 1;
	if(reverse)
	{
		printf("REVERSE!!!!");
		lstart=end-1;
		if(lstart >= reader->getNumImages())
			lstart = reader->getNumImages()-1;
		lend = start;
		linc = -1;
	}



	FullSystem* fullSystem = new FullSystem();
	fullSystem->setGammaFunction(reader->getPhotometricGamma());
	fullSystem->linearizeOperation = (playbackSpeed==0);







    IOWrap::PangolinDSOViewer* viewer = 0;
	if(!disableAllDisplay)
    {
        viewer = new IOWrap::PangolinDSOViewer(wG[0],hG[0], false);
        fullSystem->outputWrapper.push_back(viewer);
    }



    if(useSampleOutput)
        fullSystem->outputWrapper.push_back(new IOWrap::SampleOutputWrapper());



    
    // to make MacOS happy: run this in dedicated thread -- and use this one to run the GUI.
    std::thread runthread([&]() {
        std::vector<int> idsToPlay;
        std::vector<double> timesToPlayAt;
	std::vector<int> idsToPlayRight;		// right images
        std::vector<double> timesToPlayAtRight;
        for(int i=lstart;i>= 0 && i< reader->getNumImages() && linc*i < linc*lend;i+=linc)
        {
            idsToPlay.push_back(i);
            if(timesToPlayAt.size() == 0)
            {
                timesToPlayAt.push_back((double)0);
            }
            else
            {
                double tsThis = reader->getTimestamp(idsToPlay[idsToPlay.size()-1]);
                double tsPrev = reader->getTimestamp(idsToPlay[idsToPlay.size()-2]);
                timesToPlayAt.push_back(timesToPlayAt.back() +  fabs(tsThis-tsPrev)/playbackSpeed);
            }
        }
        
        for(int i=lstart;i>= 0 && i< reader_right->getNumImages() && linc*i < linc*lend;i+=linc)
        {
            idsToPlayRight.push_back(i);
            if(timesToPlayAtRight.size() == 0)
            {
                timesToPlayAtRight.push_back((double)0);
            }
            else
            {
                double tsThis = reader_right->getTimestamp(idsToPlay[idsToPlay.size()-1]);
                double tsPrev = reader_right->getTimestamp(idsToPlay[idsToPlay.size()-2]);
                timesToPlayAtRight.push_back(timesToPlayAtRight.back() +  fabs(tsThis-tsPrev)/playbackSpeed);
            }
        }

        std::vector<ImageAndExposure*> preloadedImages;
	std::vector<ImageAndExposure*> preloadedImagesRight;
        if(preload)
        {
            printf("LOADING ALL IMAGES!\n");
            for(int ii=0;ii<(int)idsToPlay.size(); ii++)
            {
                int i = idsToPlay[ii];
                preloadedImages.push_back(reader->getImage(i));
		preloadedImagesRight.push_back(reader_right->getImage(i));
            }
        }

        struct timeval tv_start;
        gettimeofday(&tv_start, NULL);
        clock_t started = clock();
        double sInitializerOffset=0;


        for(int ii=30;ii<(int)idsToPlay.size(); ii++)
        {
            if(!fullSystem->initialized)	// if not initialized: reset start time.
            {
                gettimeofday(&tv_start, NULL);
                started = clock();
                sInitializerOffset = timesToPlayAt[ii];
            }
    
            int i = idsToPlay[ii];
	    
	    double time_l = pic_time_stamp[i];
	    int index=-1;
	    if(use_stereo){
	      if(pic_time_stamp_r.size()>0){
		  for(int i=0;i<pic_time_stamp_r.size();++i){
		      if(pic_time_stamp_r[i]>=time_l||fabs(pic_time_stamp_r[i]-time_l)<0.01){
			    index = i;
			    break;					  				
		      }
		  }
	      }
	      if(fabs(pic_time_stamp_r[index]-time_l)>0.01){continue;}
	    }
// 	    LOG(INFO)<<"pic_time_stamp_r.size(): "<<pic_time_stamp_r.size()<<" pic_time_stamp.size(): "<<pic_time_stamp.size();
// 	    LOG(INFO)<<std::fixed<<std::setprecision(9)<<"time_l: "<<time_l<<" time_r: "<<pic_time_stamp_r[index];
// 	    LOG(INFO)<<"i: "<<i<<" index: "<<index;
// 	    exit(1);

            ImageAndExposure* img;
	    ImageAndExposure* img_right;
            if(preload){
                img = preloadedImages[ii];
		img_right = preloadedImagesRight[ii];
	    }
            else{
                img = reader->getImage(i);
// 		img_right = reader_right->getImage(i);
		if(use_stereo)
		  img_right = reader_right->getImage(index);
		else
		  img_right = img;
	    }



            bool skipFrame=false;
            if(playbackSpeed!=0)
            {
                struct timeval tv_now; gettimeofday(&tv_now, NULL);
                double sSinceStart = sInitializerOffset + ((tv_now.tv_sec-tv_start.tv_sec) + (tv_now.tv_usec-tv_start.tv_usec)/(1000.0f*1000.0f));

                if(sSinceStart < timesToPlayAt[ii])
                    usleep((int)((timesToPlayAt[ii]-sSinceStart)*1000*1000));
                else if(sSinceStart > timesToPlayAt[ii]+0.5+0.1*(ii%2))
                {
                    printf("SKIPFRAME %d (play at %f, now it is %f)!\n", ii, timesToPlayAt[ii], sSinceStart);
                    skipFrame=true;
                }
            }

// 	    if(i>300){
// 		imu_weight_tracker = 1;
// 	    }
// 	    if(i>600){
// 		imu_weight_tracker = 1;
// 	    }

            if(!skipFrame) fullSystem->addActiveFrame(img, img_right, i);
			  
// 	    IplImage* src = 0;
// 	    cvShowImage("camera",src);
// 	    cv::waitKey(-1);


            delete img;

            if(fullSystem->initFailed || setting_fullResetRequested)
            {
                if(ii < 250 || setting_fullResetRequested)
                {
                    printf("RESETTING!\n");

                    std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
                    delete fullSystem;

                    for(IOWrap::Output3DWrapper* ow : wraps) ow->reset();

                    fullSystem = new FullSystem();
                    fullSystem->setGammaFunction(reader->getPhotometricGamma());
                    fullSystem->linearizeOperation = (playbackSpeed==0);


                    fullSystem->outputWrapper = wraps;

                    setting_fullResetRequested=false;
		    first_track_flag = false;
                }
            }

            if(fullSystem->isLost)
            {
                    printf("LOST!!\n");
                    break;
            }

        }
        fullSystem->blockUntilMappingIsFinished();
        clock_t ended = clock();
        struct timeval tv_end;
        gettimeofday(&tv_end, NULL);


        fullSystem->printResult("result.txt");


        int numFramesProcessed = abs(idsToPlay[0]-idsToPlay.back());
        double numSecondsProcessed = fabs(reader->getTimestamp(idsToPlay[0])-reader->getTimestamp(idsToPlay.back()));
        double MilliSecondsTakenSingle = 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC);
        double MilliSecondsTakenMT = sInitializerOffset + ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
        printf("\n======================"
                "\n%d Frames (%.1f fps)"
                "\n%.2fms per frame (single core); "
                "\n%.2fms per frame (multi core); "
                "\n%.3fx (single core); "
                "\n%.3fx (multi core); "
                "\n======================\n\n",
                numFramesProcessed, numFramesProcessed/numSecondsProcessed,
                MilliSecondsTakenSingle/numFramesProcessed,
                MilliSecondsTakenMT / (float)numFramesProcessed,
                1000 / (MilliSecondsTakenSingle/numSecondsProcessed),
                1000 / (MilliSecondsTakenMT / numSecondsProcessed));
        //fullSystem->printFrameLifetimes();
        if(setting_logStuff)
        {
            std::ofstream tmlog;
            tmlog.open("logs/time.txt", std::ios::trunc | std::ios::out);
            tmlog << 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC*reader->getNumImages()) << " "
                  << ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f) / (float)reader->getNumImages() << "\n";
            tmlog.flush();
            tmlog.close();
        }

    });


    if(viewer != 0)
        viewer->run();

    runthread.join();

	for(IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper)
	{
		ow->join();
		delete ow;
	}



	printf("DELETE FULLSYSTEM!\n");
	delete fullSystem;

	printf("DELETE READER!\n");
	delete reader;

	printf("EXIT NOW!\n");
	return 0;
}
