#include <iostream>
#include "markov_localisation/map.h"
#include "markov_localisation/measurement_package.h"
#include "markov_localisation/help_functions.h"
#include "markov_localisation/bayesian_filter.h"
#include <ros/ros.h>

using namespace std;

int main() {

    string example_string = "03";

    //declare map:
    Map map_1d;

    //declare measurement list:
    std::vector<MeasurementPackage> measurement_pack_list;

    //declare helpers:
    HelpFunctions helper;

    //define file names:
    char in_file_name_ctr[1024];
    char in_file_name_obs[1024];
    char in_file_name_gt[1024];

    //read map:
    helper.ReadMapData("data/map_1d.txt", map_1d);

    std::cout << "Map read successfully..."<< std::endl;

    std::cout  <<  "Map:" << std::endl;

    for(int i=0;i<map_1d.landmark_list_.size();i++)
    {
        std::cout << map_1d.landmark_list_[i].id << " : " << map_1d.landmark_list_[i].x << std::endl;
    }


    //define file name of controls:
    sprintf(in_file_name_ctr, "data/example%s/control_data.txt",
            example_string.c_str());

    //define file name of observations:
    sprintf(in_file_name_obs, "data/example%s/observations/",
            example_string.c_str());

    //read in data to measurement package list:
    helper.ReadMeasurementData(in_file_name_ctr,
                               in_file_name_obs,
                               measurement_pack_list);

    std::cout << "Measurement data read successfully..." << std::endl;


    //create instance of 1d_bayesian localization filter:
    BayesianFilter localization_1d_bayesian;

    //define number of time steps:
    size_t T = measurement_pack_list.size();



    //cycle:
    for (size_t t = 0; t < T; ++t){

        //Call 1d_bayesian filter:
        localization_1d_bayesian.ProcessMeasurement(measurement_pack_list[t],
                                                    map_1d,
                                                    helper);
    }


    //define file name of gt data:
    sprintf(in_file_name_gt, "data/example%s/gt_example%s.txt",example_string.c_str(),example_string.c_str() );

    ///compare gt data with results:

    helper.CompareData(in_file_name_gt, localization_1d_bayesian.belief_x);

    std::cout << "Comparision done!";

    /**
      Can add publisher and subscribers.
      **/

    while(ros::ok())
    {

     ros::spinOnce();
    }


    return 0;
}
