/*
Copyright (c) 2010-2016, Mathieu Labbe - IntRoLab - Universite de Sherbrooke
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Universite de Sherbrooke nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <rtabmap/core/Odometry.h>
#include "rtabmap/core/Rtabmap.h"
#include "rtabmap/core/RtabmapThread.h"
#include "rtabmap/core/CameraRGBD.h"
#include "rtabmap/core/CameraStereo.h"
#include "rtabmap/core/CameraThread.h"
#include "rtabmap/core/OdometryThread.h"
#include "rtabmap/core/Graph.h"
#include "rtabmap/utilite/UEventsManager.h"
#include <QApplication>
#include <stdio.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/filter.h>

#include "MapBuilder.h"

void showUsage()
{
	printf("\nUsage:\n"
			"rtabmap-rgbd_mapping driver\n"
			"  driver       Driver number to use: 0=OpenNI-PCL, 1=OpenNI2, 2=Freenect, 3=OpenNI-CV, 4=OpenNI-CV-ASUS, 5=Freenect2, 6=ZED SDK, 7=RealSense, 8=RealSense2 9=Kinect for Azure SDK 10=MYNT EYE S\n\n");
	exit(1);
}

using namespace rtabmap;
int main(int argc, char * argv[])
{
	ULogger::setType(ULogger::kTypeConsole);
	ULogger::setLevel(ULogger::kWarning);
	ULogger::setLevel(ULogger::kDebug);
	

	int driver = 0;
	if(argc < 2)
	{
		// showUsage();
	}
	else
	{
		driver = atoi(argv[argc-1]);
		if(driver < 0 || driver > 10)
		{
			// UERROR("driver should be between 0 and 10.");
			// showUsage();
		}
	}

	// Here is the pipeline that we will use:
	// CameraOpenni -> "CameraEvent" -> OdometryThread -> "OdometryEvent" -> RtabmapThread -> "RtabmapEvent"

	// Create the OpenNI camera, it will send a CameraEvent at the rate specified.
	// Set transform to camera so z is up, y is left and x going forward
	Camera * camera = 0;
	if(!CameraFreenect::available())
	{
		UERROR("Not built with Freenect support...");
		exit(-1);
	}
	camera = new CameraFreenect();

	if(!camera->init())
	{
		UERROR("Camera init failed!");
	}

	int odomUpdate = 20;
	int mapUpdate = 1;
	Odometry * odom = Odometry::create();
	Rtabmap rtabmap;
	rtabmap.init();

	QApplication app(argc, argv);
	MapBuilder mapBuilder;
	mapBuilder.show();

	QApplication::processEvents();
	SensorData data = camera->takeImage();

	printf("Press \"Space\" in the window to pause\n");
	int cameraIteration = 0;
	int odometryIteration = 0;	
	while(data.isValid() && mapBuilder.isVisible())
	{
		if(cameraIteration++ % odomUpdate == 0)
		{
			OdometryInfo info;
			Transform pose = odom->process(data, &info);
			if(odometryIteration++ % mapUpdate == 0)
			{
				if(rtabmap.process(data, pose))
				{
					mapBuilder.processStatistics(rtabmap.getStatistics());
					if(rtabmap.getLoopClosureId() > 0)
					{
						printf("Loop closure detected!\n");
					}
				}
			}
			mapBuilder.processOdometry(data, pose, info);
		}
		QApplication::processEvents();
		while(mapBuilder.isPaused() && mapBuilder.isVisible())
		{
			uSleep(100);
			QApplication::processEvents();
		}
		data = camera->takeImage();
	}	
	return 0;
}
