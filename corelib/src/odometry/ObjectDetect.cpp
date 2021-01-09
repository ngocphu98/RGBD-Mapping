#include "ObjectDetect.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h> 
#include "opencv2/core.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <unistd.h>
#include "rtabmap/utilite/ULogger.h"
#include <rtabmap/core/util3d.h>
#include <rtabmap/core/CameraModel.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/pcl_base.h>
#include <pcl/TextureMesh.h>

#define nominator(_xa, _xb, _xc, _ya, _yb, _yc, _da, _db, _dc) ((  pow((float)_xc,2) - pow((float)_xa,2)\
                                                                 + pow((float)_yc,2) - pow((float)_ya,2)\
                                                                 + pow((float)_da,2) - pow((float)_dc,2))\
                                                                 * (_yc - _yb)\
                                                                 -(pow((float)_xc,2) - pow((float)_xb,2)\
                                                                 + pow((float)_yc,2) - pow((float)_yb,2)\
                                                                 + pow((float)_db,2) - pow((float)_dc,2))\
                                                                 * (_yc - _ya))
                    
#define denominator(_xa, _xb, _xc, _ya, _yb, _yc) ((2*(_xc - _xa)*(_yc - _yb)) - (2*(_xc - _xb)*(_yc - _ya)))


ObjDetect::ObjDetect(std::string objPath)
{
    loadAllObject(objPath);
}

bool ObjDetect::loadAllObject(std::string objPath)
{
    std::vector<cv::String> images;
    cv::glob(objPath + "/*.jpg", images, false);
    _objCount = images.size(); //number of png files in images folder
    //open file Depths.txt
    std::string line;
    std::ifstream DepthsFile(objPath + "/Depths.txt");
    if (_objCount <= MAX_OBJECT)
    {
        for (size_t i = 0; i < _objCount; i++)
        {
            _objects[i].grayImg = (cv::imread(images[i])); 
            std::getline(DepthsFile, line);
            size_t pos = 0;
            pos = line.find(' ');
            //get x
            std::string s_tmp = line.substr(0, pos);
            _objects[i].coordinate.x = strtof(s_tmp.c_str(), NULL);
            //get y
            s_tmp = line.substr(pos, line.length());
            _objects[i].coordinate.y = strtof(s_tmp.c_str(), NULL);
            #ifdef DEBUG
            UDEBUG("information of object: i= %d, path= %s, x= %d, y= %d", i, images[i], _objects[i].coordinate.x, _objects[i].coordinate.y);
            #endif /*DEBUG*/
            _detector->detectAndCompute(_objects[i].grayImg, 
                                        cv::noArray(),
                                        _objects[i].keyPoints,
                                        _objects[i].descriptor
                                        );
        }
    }
    else
    {
        UERROR("Too much object Maximum object is:");
        return false;
    }

    return true;
}

std::vector<cv::DMatch> ObjDetect::filterMatched(std::vector< std::vector<cv::DMatch> > knn_matches)
{
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.75f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }    
    return good_matches;
}

cv::Point2f ObjDetect::detectObject(Object data)
{
    int matched_object = 0;
    objectInfor obj_id1;
    objectInfor obj_id2;
    objectInfor obj_id3;

    _detector->detectAndCompute(data.grayImg, 
                                cv::noArray(),
                                data.keyPoints,
                                data.descriptor
                                );      

    //Loop all Object and find 3 matched object
    for (int i = 0; i < _objCount; i++)
    {
        std::vector< std::vector<cv::DMatch> > knn_matches;
        _matcher->knnMatch(_objects[i].descriptor, data.descriptor, knn_matches, 2);
        std::vector<cv::DMatch> good_matches;
        good_matches = this->filterMatched(knn_matches);
        #ifdef DEBUG
        int size_tmp = good_matches.size();
        UDEBUG("good matched point founded: %d",size_tmp);
        #endif
        if (good_matches.size() >= 10)
        {
            matched_object++;
            if (matched_object == 1)
            {
                obj_id1.id = i;
                obj_id1.distance = this->getDepth(data, _objects[i], good_matches);
                if(isnan(obj_id1.distance))
                {
                    matched_object--;
                }
                UDEBUG("obj_id1 distance: %f", obj_id1.distance);
            }
            if (matched_object == 2)
            {
                obj_id2.id = i;
                obj_id2.distance = this->getDepth(data, _objects[i], good_matches);
                if(isnan(obj_id2.distance))
                {
                    matched_object--;
                }                
                UDEBUG("obj_id2 distance: %f", obj_id2.distance);
            }
            if (matched_object == 3)
            {
                obj_id3.id = i;
                obj_id3.distance = this->getDepth(data, _objects[i], good_matches);
                if(isnan(obj_id3.distance))
                {
                    matched_object--;
                }                
                UDEBUG("obj_id3 distance: %f", obj_id3.distance);
            }            
            if (matched_object >= 3)
            {
                matched_object = 0;
                // cv::waitKey();
                return this->ComputeCoordinate(obj_id1, obj_id2, obj_id3);
                break;
            }
            #ifdef DEBUG
            //-- Maping data to new variable. Avoid change variable in code blow
            cv::Mat img_scene = data.grayImg;
            cv::Mat img_object = _objects[i].grayImg;
            std::vector<cv::KeyPoint> keypoints_object = _objects[i].keyPoints;
            std::vector<cv::KeyPoint> keypoints_scene = data.keyPoints;
            //-- Draw matches
            cv::Mat img_matches;
            cv::drawMatches( img_object, keypoints_object, img_scene, keypoints_scene, good_matches, img_matches, cv::Scalar::all(-1),
                         cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
            //-- Localize the object
            std::vector<cv::Point2f> obj;
            std::vector<cv::Point2f> scene;            
            for( size_t i = 0; i < good_matches.size(); i++ )
            {
                //-- Get the keypoints from the good matches
                obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
                scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
            }
            cv::Mat H = cv::findHomography( obj, scene, cv::RANSAC );
            //-- Get the corners from the image_1 ( the object to be "detected" )
            std::vector<cv::Point2f> obj_corners(4);
            obj_corners[0] = cv::Point2f(0, 0);
            obj_corners[1] = cv::Point2f( (float)img_object.cols, 0 );
            obj_corners[2] = cv::Point2f( (float)img_object.cols, (float)img_object.rows );
            obj_corners[3] = cv::Point2f( 0, (float)img_object.rows );
            std::vector<cv::Point2f> scene_corners(4);
            if (!H.empty())
            {
                cv::perspectiveTransform( obj_corners, scene_corners, H);
                //-- Draw lines between the corners (the mapped object in the scene - image_2 )
                cv::line( img_matches, scene_corners[0] + cv::Point2f((float)img_object.cols, 0),
                      scene_corners[1] + cv::Point2f((float)img_object.cols, 0), cv::Scalar(0, 255, 0), 4 );
                cv::line( img_matches, scene_corners[1] + cv::Point2f((float)img_object.cols, 0),
                      scene_corners[2] + cv::Point2f((float)img_object.cols, 0), cv::Scalar( 0, 255, 0), 4 );
                cv::line( img_matches, scene_corners[2] + cv::Point2f((float)img_object.cols, 0),
                      scene_corners[3] + cv::Point2f((float)img_object.cols, 0), cv::Scalar( 0, 255, 0), 4 );
                cv::line( img_matches, scene_corners[3] + cv::Point2f((float)img_object.cols, 0),
                      scene_corners[0] + cv::Point2f((float)img_object.cols, 0), cv::Scalar( 0, 255, 0), 4 );
                //-- Show detected matches
                cv::imshow("Good Matches & Object detection", img_matches );
                // cv::waitKey();            
            }
            #endif /* #ifdef DEBUG*/
        }
    }
    return cv::Point2i(NULL,NULL);
    
}


cv::Point2f ObjDetect::ComputeCoordinate(objectInfor objectId1, objectInfor objectId2, objectInfor objectId3)
{   
    UDEBUG("Objects1.coordinate.x: = %f, y = %f", _objects[objectId1.id].coordinate.x, _objects[objectId1.id].coordinate.y);
    UDEBUG("Objects2.coordinate.x: = %f, y = %f", _objects[objectId2.id].coordinate.x, _objects[objectId2.id].coordinate.y);
    UDEBUG("Objects3.coordinate.x: = %f, y = %f", _objects[objectId3.id].coordinate.x, _objects[objectId3.id].coordinate.y);

    float xa = _objects[objectId1.id].coordinate.x;
    float ya = _objects[objectId1.id].coordinate.y; 
    float xb = _objects[objectId2.id].coordinate.x;
    float yb = _objects[objectId2.id].coordinate.y;
    float xc = _objects[objectId3.id].coordinate.x;
    float yc = _objects[objectId3.id].coordinate.y;    
    float da = objectId1.distance;
    float db = objectId2.distance;
    float dc = objectId3.distance;
    cv::Point2f coordinate = cv::Point2f(0, 0);
    coordinate.x = nominator(xa, xb, xc, ya, yb, yc, da, db, dc)/denominator(xa,xb,xc, ya, yb, yc);
    coordinate.y = nominator(xb, xa, xc, yb, ya, yc, db, da, dc)/denominator(xa,xb,xc, ya, yb, yc);
    return coordinate;
}

float ObjDetect::getDepth(Object data, Object object, std::vector<cv::DMatch> good_matches)
{
    //-- Maping data to new variable. Avoid change variable in code blow
    cv::Mat img_scene = data.grayImg;
    cv::Mat img_object = object.grayImg;
    std::vector<cv::KeyPoint> keypoints_object = object.keyPoints;
    std::vector<cv::KeyPoint> keypoints_scene = data.keyPoints;
    //-- Draw matches
    cv::Mat img_matches;
    cv::drawMatches( img_object, keypoints_object, img_scene, keypoints_scene, good_matches, img_matches, cv::Scalar::all(-1),
                 cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //-- Localize the object
    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;            
    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }
    cv::Mat H = cv::findHomography( obj, scene, cv::RANSAC );
    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<cv::Point2f> obj_corners(4);
    obj_corners[0] = cv::Point2f(0, 0);
    obj_corners[1] = cv::Point2f( (float)img_object.cols, 0 );
    obj_corners[2] = cv::Point2f( (float)img_object.cols, (float)img_object.rows );
    obj_corners[3] = cv::Point2f( 0, (float)img_object.rows );
    std::vector<cv::Point2f> scene_corners(4);
    if (!H.empty())
    {
        cv::perspectiveTransform( obj_corners, scene_corners, H);
        #ifdef DEBUG
        //test drawing object
        cv::line( img_scene, scene_corners[0], scene_corners[1], cv::Scalar(0, 255, 0),  4 );
        cv::line( img_scene, scene_corners[1], scene_corners[2], cv::Scalar( 0, 255, 0), 4 );
        cv::line( img_scene, scene_corners[2], scene_corners[3], cv::Scalar( 0, 255, 0), 4 );
        cv::line( img_scene, scene_corners[3], scene_corners[0], cv::Scalar( 0, 255, 0), 4 );  
        //Draw Degonal line of object
        cv::line( img_scene, scene_corners[0], scene_corners[2], cv::Scalar(0, 255, 0), 4);      
        cv::line( img_scene, scene_corners[1], scene_corners[3], cv::Scalar(0, 255, 0), 4);
        #endif //#ifdef DEBUG
        cv::Point2f centralPoint = cv::Point2f((scene_corners[1].x - scene_corners[0].x)/2 + scene_corners[0].x,
            (scene_corners[1].y - scene_corners[2].y)/2 + scene_corners[2].y);
        #ifdef DEBUG
        UDEBUG("Central Point: x = %d, y = %d", centralPoint.x, centralPoint.y);  
        cv::circle(img_scene, centralPoint, 2, cv::Scalar(255, 255, 255), 2);
        cv::imshow("Good Matches & Object detection", img_scene );
    #endif //#ifdef DEBUG
        if ((centralPoint.x <=data.depth.cols) && (centralPoint.y <=data.depth.rows))
            return rtabmap::util3d::cloudFromDepth(data.depth, 319.5, 239.5, 525.0, 525)->at(round(centralPoint.x), round(centralPoint.y)).z;
    }

    return 0;
}

