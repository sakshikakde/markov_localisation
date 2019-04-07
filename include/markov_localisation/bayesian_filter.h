/*
 * basian_filter.h
 *
 *  Created on: Apr 26, 2017
 *      Author: ramiz
 */

#ifndef BAYESIAN_FILTER_H_
#define BAYESIAN_FILTER_H_

#include <vector>
#include <string>
#include <fstream>
#include <bits/stdc++.h>

#include "markov_localisation/measurement_package.h"
#include "markov_localisation/map.h"
#include "markov_localisation/help_functions.h"

class BayesianFilter {
public:
  //constructor
  BayesianFilter();

  //destructor
  virtual ~BayesianFilter();

  //main public member function, which estimate the beliefs:
  void ProcessMeasurement(const MeasurementPackage &measurements,
                         const Map &map_1d,
                         HelpFunctions &helpers);

  std::vector<float> belief_x;

private:
  /**
   * Initializes the initial belief
   * @param map_1d, 1D map of all landmarks
   * @helpers, helper functions class reference
   */
  void Init(const Map &map_1d, HelpFunctions &helpers);
  /**
   * Applies motion model to calculate prior Xt (predict Xt)
   * @param measurements, list of measurements (controls, observations)
   * @param map_1d, 1D map of all landmarks
   * @helpers, helper functions class reference
   */
  float Predict(int i, const MeasurementPackage &measurements,
               const Map &map_1d,
               HelpFunctions &helpers);

  /**
   * Applies observation model to calculate likelihood of current observation Zt for Xt predicted
   * @param measurements, list of measurements (controls, observations)
   * @param map_1d, 1D map of all landmarks
   * @helpers, helper functions class reference
   */
  float Update(int i, const MeasurementPackage &measurements,
               const Map &map_1d,
               HelpFunctions &helpers);

  bool is_initialized_;
  std::vector<float> belief_x_init_;
  float control_std_;
  float observation_std_;

  std::ofstream posterior_control_file_, posterior_obs_file_, final_bell_file_;
};



#endif /* BAYESIAN_FILTER_H_ */
