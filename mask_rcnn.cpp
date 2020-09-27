#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace dnn;


float thres = 0.7; //confidence threshold
float maskThres = 0.4; //mask threshold
vector<string> labels;
vector<Scalar> colors;


void setColor(){
    //color to distinct each instance
    colors.push_back(Scalar(0,255,0,255.0));
    colors.push_back(Scalar(0,0,255,255.0));
    colors.push_back(Scalar(255,0,0,255.0));
    colors.push_back(Scalar(0,255,255,255.0));
    colors.push_back(Scalar(255,0,255,255.0));
    colors.push_back(Scalar(100,200,130,255.0));
    colors.push_back(Scalar(50,60,140,255.0));
    colors.push_back(Scalar(130,110,180,255.0));
    colors.push_back(Scalar(170,175,230,255.0));
    colors.push_back(Scalar(80,90,200,255.0));
}


vector<string> getLabel(string file_name){
    vector<string>label;
    ifstream inputStream(file_name.c_str());
    string read_line_by_line;
    while(getline(inputStream, read_line_by_line)){
        label.push_back(read_line_by_line);
    }
    return label;
}


String graph_file = "./mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"; // graph
String model_weight = "./mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"; //pre trained weight
string labelFile = "mscoco_labels.names"; //load label names from a text file

int main(){
    string image_path, outputFile;
    vector<Mat> get_output_from_the_network;
    vector<String>output_layer;
    String name_of_output_layer[2];
    Mat image, blob_image;
    Size size_of_label;
    Mat resulted_image;

    image_path= "chemnitz.jpg"; //"cars.jpg";//
    labels=getLabel(labelFile);
    setColor();
    Net net = readNetFromTensorflow(model_weight, graph_file); //Reads a network model stored in TensorFlow framework's format
    /*
        https://docs.opencv.org/3.4/d6/d0f/group__dnn.html#gad820b280978d06773234ba6841e77e8d
        readNetFromTensorflow(model,config)
        model  - path to the .pb file with binary protobuf description of the network architecture
        config - path to the .pbtxt file that contains text graph definition in protobuf format.

    */
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU); //run on CPU

    outputFile="output/sample_output.jpeg"; //resulted path
    image=imread(image_path); //read image from the given path

    blobFromImage(image,blob_image,1.0,Size(image.cols, image.rows),Scalar(),true, false); //Image to BLOB
    /*
        https://docs.opencv.org/3.4/d6/d0f/group__dnn.html#ga29f34df9376379a603acd8df581ac8d7
        cv.dnn.blobFromImage(image[, scalefactor[, size[, mean[, swapRB[, crop[, ddepth]]]]]])
        BLOB is Binary Large OBject. A group of connected pixels in a binary image
    */


    net.setInput(blob_image); //feed to the network


    name_of_output_layer[0] = "detection_out_final"; //this layer is responsible for predicting bounding box of the object
    name_of_output_layer[1] = "detection_masks"; // this layer is responsible for predicting the mask of the object
    output_layer.push_back(name_of_output_layer[0]);
    output_layer.push_back(name_of_output_layer[1]);

    net.forward(get_output_from_the_network, output_layer); //forward pass
    /*
        https://docs.opencv.org/3.4/db/d30/classcv_1_1dnn_1_1Net.html#adb34d7650e555264c7da3b47d967311b

        after forward pass (actually inference is done here),
        we get two things: 1. detected object 2. mask of the object
    */


    Mat detected_object = get_output_from_the_network[0]; //take the detected object
    /*
        Here,
        detected_object[i][1] = class ID (label)
        detected_object[i][2] = confidence score for that class (probability)
        detected_object[i][3] = x1 (coordinate or shape information are extracted)
        detected_object[i][4] = y1
        detected_object[i][5] = x2
        detected_object[i][6] = y2
    */

    Mat masks = get_output_from_the_network[1]; // take the mask
    /*
        Here,
        masks[0] = the number of detected boxes in the frame
        masks[1] = the number of classes in the model
        masks[2] = width
        masks[3] = height
    */

    int num_of_detected_object = detected_object.size[2];
    int number_of_detected_class = masks.size[1];

    detected_object = detected_object.reshape(1, detected_object.total()/detected_object.size[3]);
    for(int i=0;i<num_of_detected_object;i++){
        cout<<"Test "<<detected_object.at<float>(i, 2)<<endl;
        int id = detected_object.at<float>(i, 1); //extract the class ID
        float confidence = detected_object.at<float>(i,2); //extract confidence (probability) that is associate with the prediction
        if (confidence>thres){
            //coordinate extraction to form the bounding box
            int x1 = image.cols*detected_object.at<float>(i,3); //extract x1
            int y1 = image.rows*detected_object.at<float>(i,4);
            int x2 = image.cols*detected_object.at<float>(i,5);
            int y2 = image.rows*detected_object.at<float>(i,6);
            Rect bounding_box = Rect(x1,y1,x2-x1,y2-y1); //define bounding box
            Mat extracted_mask(masks.size[2], masks.size[3],CV_32F, masks.ptr<float>(i,id)); //the mask is extracted for the detected object
            rectangle(image, Point(bounding_box.x, bounding_box.y), Point(bounding_box.x+bounding_box.width, bounding_box.y+bounding_box.height), Scalar(0,0,255),2);
            string labelOfTheDetectedObject= labels[id] + "=" + to_string(confidence);
            //Display the label at the top of the bounding box
            int base_line=0;
            size_of_label = getTextSize(labelOfTheDetectedObject,FONT_HERSHEY_SIMPLEX,0.6,1.2,&base_line);
            /*
                getTextSize(text, fontFace, fontScale, thickness, &baseline);
            */

            rectangle(image,Point(bounding_box.x,bounding_box.y-size_of_label.height),Point(bounding_box.x+size_of_label.width, bounding_box.y + base_line), Scalar(255, 255, 255), FILLED);
            putText(image, labelOfTheDetectedObject, Point(bounding_box.x, bounding_box.y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1); //show label
            Scalar color = colors[id%colors.size()]; //color for that particular class (id)
            resize(extracted_mask,extracted_mask,Size(bounding_box.width,bounding_box.height));
            Mat mask=extracted_mask>=maskThres; // only considered those mask value that are equal or greater than the mask threshold
            Mat ROI = color+image(bounding_box);
            ROI.convertTo(ROI, CV_8UC3); //CV_8UC3 is an 8-bit unsigned integer matrix/image with 3 channels
            vector<Mat> distinct_instance; //every instance of the object is identified
            Mat hierarchy;
            mask.convertTo(mask, CV_8U); //CV_8U is unsigned 8bit pixel
            findContours(mask, distinct_instance, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
            drawContours(ROI, distinct_instance,-1,color,5,LINE_8,hierarchy,100);
            ROI.copyTo(image(bounding_box),mask);
            /*
                copyTo(OutputArray m, InputArray mask)
                here, m – Destination matrix
                      mask – Operation mask
            */

        }

    }

    image.convertTo(resulted_image, CV_8U);
    imwrite(outputFile, resulted_image); //store the resulted image

    return 0;
}

