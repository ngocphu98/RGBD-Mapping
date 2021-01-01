#include "ObjectDetect.h"
#include <iostream>
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

#define nominator(xa, xb, xc, da, db, dc) ((pow((float)xc,(float)2) - pow((float)xa,(float)2) \
                            + pow((float)yc,(float)2) - pow((float)ya,(float)2)\
                            + pow((float)da,(float)2) - pow((float)dc,(float)2))\
                            * (yc - yb)\
                            - (pow((float)xc,(float)2) - pow((float)xb,(float)2) \
                            + pow((float)yc,(float)2) - pow((float)yb,(float)2)\
                            + pow((float)db,(float)2) - pow((float)dc,(float)2))\
                            * (yc - ya))
                    
#define denominator(xa, xb, xc) (2*(xc-xa)*(yc-yb) -2*(xc-xb)*(yc-ya))


ObjDetect::ObjDetect(std::string objPath)
{
    loadAllObject(objPath);
}

bool ObjDetect::loadAllObject(std::string objPath)
{
    std::vector<cv::String> images;
    cv::glob(objPath, images, false);
    _objCount = images.size(); //number of png files in images folder
    if (_objCount <= MAX_OBJECT)
    {
        for (size_t i = 0; i < _objCount; i++)
            {
            _objects[i].grayImg = (cv::imread(images[i]));  
            #ifdef DEBUG
            std::cout << "information of object: "; std::cout << i; std::cout << "\n";
            std::cout << "Path: "; std::cout << images[i]; std::cout << "\n";
            std::cout << "Rows: "; std::cout <<_objects[i].grayImg.rows; std::cout << "\n";
            std::cout << "Cols: "; std::cout <<_objects[i].grayImg.cols; std::cout << "\n";             
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
        std::cout <<"Too much object Maximum object is:"; std::cout << MAX_OBJECT;
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

cv::Point2i ObjDetect::detectObject(Object data)
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
        std::cout << "good matched point founded: " << good_matches.size() <<"\n";
        #endif
        if (good_matches.size() >= 5)
        {
            matched_object++;
            if (matched_object == 1)
            {
                obj_id1.id = i;
                obj_id1.distance = this->getDepth(data, _objects[i], good_matches);
            }
            if (matched_object == 2)
            {
                obj_id2.id = i;
                obj_id2.distance = this->getDepth(data, _objects[i], good_matches);
            }
            if (matched_object == 3)
            {
                obj_id3.id = i;
                std::cout << "matched_object: " <<matched_object;
                std::cout << "index: " << i;
                obj_id3.distance = this->getDepth(data, _objects[i], good_matches);
            }            
            if (matched_object >= 3)
            {
                std::cout << "\nPhu is Here!\n";
                std::cout << "\nobj_id1: " << obj_id1.distance;
                std::cout << "\nobj_id2: " << obj_id2.distance;
                std::cout << "\nobj_id3: " << obj_id3.distance;
                std::cout << "\nobj_id3: " << obj_id3.id;
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
                std::cout <<"Obj_corners: "<< obj_corners.size() << '\n';
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
    return cv::Point2i(0,0);
    
}


cv::Point2i ObjDetect::ComputeCoordinate(objectInfor objectId1, objectInfor objectId2, objectInfor objectId3)
{   
    std::cout << "Enter to ComputeCoordinate ^_^";
    std::cout << "_objects[objectId1.id].coordinate.x: " << _objects[objectId1.id].coordinate.x;
    std::cout << "_objects[objectId1.id].coordinate.y: " << _objects[objectId1.id].coordinate.y;    
    std::cout << "_objects[objectId2.id].coordinate.x: " << _objects[objectId2.id].coordinate.x;
    std::cout << "_objects[objectId2.id].coordinate.y: " << _objects[objectId2.id].coordinate.y;    
    // std::cout << "_objects[objectId3.id].coordinate.x: " << _objects[objectId3.id].coordinate.x;
    // std::cout << "_objects[objectId3.id].coordinate.y: " << _objects[objectId3.id].coordinate.y;
    // cv::waitKey();
    int xa = _objects[objectId1.id].coordinate.x;
    int ya = _objects[objectId1.id].coordinate.y; 
    int xb = _objects[objectId2.id].coordinate.x;
    int yb = _objects[objectId2.id].coordinate.y;
    int xc = _objects[objectId3.id].coordinate.x;
    int yc = _objects[objectId3.id].coordinate.y;    
    int da = objectId1.distance;
    int db = objectId2.distance;
    int dc = objectId3.distance;
    cv::Point2i coordinate = cv::Point2i(0, 0);
    coordinate.x = nominator(xa, xb, xc, da, db, dc)/denominator(xa,xb,xc);
    coordinate.y = nominator(xb, xa, xc, db, da, dc)/denominator(xa,xb,xc); 
    std::cout << "coordinate.x" <<coordinate.x;
    std::cout << "coordinate.y" <<coordinate.y;
    return coordinate;
}

int ObjDetect::getDepth(Object data, Object object, std::vector<cv::DMatch> good_matches)
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
        std::cout <<"Obj_corners in getDept(): "<< obj_corners.size() << '\n';    
        // usleep((useconds_t)3000);
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
        std::cout <<"Corner 0: " <<scene_corners[0].x <<' ' <<scene_corners[0].y << '\n';      
        std::cout <<"Corner 1: " <<scene_corners[1].x <<' ' <<scene_corners[1].y << '\n';      
        std::cout <<"Corner 2: " <<scene_corners[2].x <<' ' <<scene_corners[2].y << '\n';      
        std::cout <<"Corner 3: " <<scene_corners[3].x <<' ' <<scene_corners[3].y << '\n';      
        #endif //#ifdef DEBUG
        cv::Point2f centralPoint = cv::Point2f((scene_corners[1].x - scene_corners[0].x)/2 + scene_corners[0].x,
            (scene_corners[1].y - scene_corners[2].y)/2 + scene_corners[2].y);
        #ifdef DEBUG
        std::cout <<"x = " << centralPoint.x << '\n';
        std::cout <<"y = " << centralPoint.y << '\n';
        cv::circle(img_scene, centralPoint, 2, cv::Scalar(255, 255, 255), 2);
        //-- Show detected matches
        cv::imshow("Good Matches & Object detection", img_scene );
        // cv::waitKey();    
    #endif //#ifdef DEBUG
        std::cout << "\ndata.depth.cols " << data.depth.cols;
        std::cout << "\ndata.depth.rows " << data.depth.rows;
        if ((centralPoint.x <=data.depth.cols) && (centralPoint.y <=data.depth.rows))
            return data.depth.at<int>(cv::Point(round(centralPoint.x), round(centralPoint.y)));
    }

    return 0;
}

