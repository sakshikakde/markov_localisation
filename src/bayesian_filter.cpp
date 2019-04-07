#include "markov_localisation/bayesian_filter.h"
#include <iostream>
#include <fstream>

BayesianFilter::BayesianFilter() {
    is_initialized_ = false;
    control_std_ = 1.0f;
    observation_std_ = 1.0f;

    belief_x_init_.resize(100, 0);
    belief_x.resize(100, 0);

    posterior_control_file_.open("post_controll.txt");
    posterior_obs_file_.open("post_obs.txt");
    final_bell_file_.open("bell_dist.txt");


}

BayesianFilter::~BayesianFilter() {

    posterior_control_file_.close();
    posterior_obs_file_.close();
    final_bell_file_.close();
    std::cout <<  "file closed!";

}

void BayesianFilter::Init(const Map &map_1d, HelpFunctions &helpers) {
    //run over map, all landmark values in map_1d

    for (int l = 0; l < map_1d.landmark_list_.size(); ++l) {
        //get landmark l from map
        Map::SingleLandmark single_landmark = map_1d.landmark_list_[l];

        //check if landmark position x fits in map[0, 100]
        if (!(single_landmark.x >= 0 && single_landmark.x < 100)) {
            continue;
        }

        //set belief to 1 at position and +/- from position:
        int x = single_landmark.x;
        belief_x_init_[x] = 1;
        if (x > 0) {
            belief_x_init_[x - 1] = 1;
        }

        if (x < 99) {
            belief_x_init_[x + 1] = 1;
        }
    } //end for loop

    //normalize initial believe

    std::cout << "initial belief" << std::endl;
    belief_x_init_ = helpers.NormalizeVector(belief_x_init_);


    is_initialized_ = true;
}

float BayesianFilter::Predict(int i, const MeasurementPackage &measurements,
                              const Map &map_1d, HelpFunctions &helpers) {
    /******************************************************************************
   *  motion model update
   ******************************************************************************/
    //  std::cout << "-->motion model for state x ! \n" << std::endl;

    //get current observations and control information:
    MeasurementPackage::Control controls = measurements.control_;
    float pose_i = float(i);

    /**************************************************************************
   *  posterior for motion model
   **************************************************************************/

    // motion posterior:
    // used to set up the convlution
    float posterior_motion = 0.0f;

    //loop over state space x_t-1 * same size as bel_x (Perform Convolution):
    for (int j = 0; j < belief_x_init_.size(); ++j) {

        float pose_j = float(j);
        float distance_ij = pose_i - pose_j;
        //Calculate transition probabilites using helpers.normpdf()
        // x: difference between bel_x index and state space index
        // mu: the movement from controls defined above
        // std: defined eariler
        float transition_model_probability = helpers.Normpdf(distance_ij,
                                                             controls.delta_x, control_std_);

        //Calculate motion model
        // ADD the transition prob multiplied by the initial believe
        // at state space index

        posterior_motion += transition_model_probability * belief_x_init_[j];
    }
    return posterior_motion;
}

float BayesianFilter::Update(int i, const MeasurementPackage &measurements,
                             const Map &map_1d, HelpFunctions &helpers) {

    MeasurementPackage::Observation observations = measurements.observation_;

    /**************************************************************************
   *  observation update:
   **************************************************************************/
    float pose_i = float(i);
    //define pseudo observation vector:
    std::vector<float> pseudo_ranges;

    //define maximum distance:
    float distance_max = 100;

    //loop over number of landmarks and estimate pseudo ranges:
    for (int l = 0; l < map_1d.landmark_list_.size(); ++l) {

        //calculate difference between landmark position
        // and current believe state index
        float range_l = map_1d.landmark_list_[l].x - pose_i;

        //check, if distances are positive, and store positive range:
        if (range_l >= 0.0f) {
            pseudo_ranges.push_back(range_l);
        }
    }

    //sort pseudo range vector:
    sort(pseudo_ranges.begin(), pseudo_ranges.end());

    //define observation posterior:
    float posterior_obs = 1.0f;

    //run over current observations vector defined above:
    for (int z = 0; z < observations.distances.size(); ++z) {

        //define min distance:
        float pseudo_range_min_distance;

        // set min distance either to the closet landmark
        // or if no landmarks exist to the maximum set distance
        if (pseudo_ranges.size() > 0) {
            //set min distance
            pseudo_range_min_distance = pseudo_ranges[0];
            //remove this entry
            pseudo_ranges.erase(pseudo_ranges.begin());
        } else {
            pseudo_range_min_distance = distance_max;
        }

        //estimate the posterior for observation model:
        // MULTIPLY by normpdf of obseravations distance,
        // min distance, and obseravtion_std


        posterior_obs *= helpers.Normpdf(observations.distances[z],
                                         pseudo_range_min_distance, observation_std_);
    }

    return posterior_obs;
}

void BayesianFilter::ProcessMeasurement(const MeasurementPackage &measurements,
                                        const Map &map_1d,
                                        HelpFunctions &helpers) {



    /******************************************************************************
   *  Set init belief of state vector:
   ******************************************************************************/
    if (!is_initialized_) {
        Init(map_1d, helpers);
    } //end if


    std::cout << std::endl<<"control :" << measurements.control_.delta_x << "\t" <<"observation :";//<< measurements.observation_.distances << std::endl;

    for(int j=0;j<measurements.observation_.distances.size();j++)
    {
        std::cout << measurements.observation_.distances.at(j) << "\t";
    }



    for (int i = 0; i < belief_x.size(); i++) {
        /******************************************************************************
     *  motion model update
     ******************************************************************************/
        float posterior_motion = Predict(i, measurements, map_1d, helpers);

        posterior_control_file_<< posterior_motion << "\t" ;
        /******************************************************************************
     *  observation model update
     ******************************************************************************/
        float posterior_observation = Update(i, measurements, map_1d, helpers);

        posterior_obs_file_<< posterior_observation << "\t" ;

        //update the belief for Xt_i
        belief_x[i] = posterior_observation * posterior_motion;
    }



    posterior_control_file_ << "\n";
    posterior_obs_file_ << "\n";

    //normalize bel_x so that sum of all probabilities equals 1
    belief_x = helpers.NormalizeVector(belief_x);

    std::cout <<"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    for(int k=0;k<belief_x_init_.size();k++)
    {
        std::cout << belief_x_init_.at(k) << std::endl;
        final_bell_file_ << belief_x_init_.at(k);
        final_bell_file_ << "\t";
    }

    final_bell_file_<<"\n" ;
    std::cout <<">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;

    //set initial believe to bel_x for next time
    belief_x_init_ = belief_x;
}
