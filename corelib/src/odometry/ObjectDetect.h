#ifndef OBJECTDETECT_H_
#define OBJECTDETEC_H_
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

// #include "rtabmap/core/SensorData.h"

#ifndef DEBUG
#define DEBUG
#endif

#define MAX_OBJECT 10



struct Object
{
    cv::Mat grayImg;
    cv::Mat depth;
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptor;
    cv::Point2i coordinate = cv::Point2i(0,0);    
};

struct objectInfor
{
    int id;
    int distance;
};

class ObjDetect
{
    public:
        int objectId = 0;
        int _minHessian = 400;
        cv::Ptr<cv::xfeatures2d::SURF> _detector = cv::xfeatures2d::SURF::create( _minHessian );         
        cv::Ptr<cv::DescriptorMatcher> _matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    public:
        ObjDetect(std::string dbPath);
        //init for ObjDetect
        //Load all the object from the data base
        //Return 1 to suscess load
        bool loadAllObject(std::string objPath);
        cv::Point2i detectObject(Object data);
        cv::Point2i ComputeCoordinate(objectInfor objectId1, objectInfor objectId2, objectInfor objectId3);
        int getDepth(Object data, Object img_object, std::vector<cv::DMatch> good_matches);

        // coordinatesType detectObject(SensorData data);
        // coordinatesType ComputeLocalization(SensorData data);

    private:
        size_t _objCount;
        Object _objects[MAX_OBJECT] = {};
        std::vector<cv::DMatch> filterMatched(std::vector< std::vector<cv::DMatch> > knn_matches);


};

#endif /*OBJECTDETECT_H_*/