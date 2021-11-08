//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"

/// gc_ = x,y,z positions
///       w,x,y,z quaternion
///       w1,w2,w3,w4 wheel angles

/// gw_ = x,y,z linear velocities
///       w_x,w_y,w_z angular velocities
///       s1,s2,s3,s4 wheel speed

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

    /// create world
    world_ = std::make_unique<raisim::World>();

    /// add robot
    husky_ = world_->addArticulatedSystem(resourceDir_+"/husky/husky.urdf");
    husky_->setName("husky");
    husky_->setControlMode(raisim::ControlMode::FORCE_AND_TORQUE);

    /// add heightmap
    raisim::TerrainProperties terrainProperties;
    terrainProperties.frequency = 0.2;
    terrainProperties.zScale = 0.1;
    terrainProperties.xSize = 70.0;
    terrainProperties.ySize = 70.0;
    terrainProperties.xSamples = 70;
    terrainProperties.ySamples = 70;
    terrainProperties.fractalOctaves = 3;
    terrainProperties.fractalLacunarity = 2.0;
    terrainProperties.fractalGain = 0.25;
    heightMap_ = world_->addHeightMap(0.0, 0.0, terrainProperties);

    /// get robot data
    gcDim_ = husky_->getGeneralizedCoordinateDim();
    gvDim_ = husky_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    genForce_.setZero(gcDim_); torque4_.setZero(nJoints_);

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 17;
    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(10.);

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(husky_);
    }
  }

  void init() final { }

  void reset() final {
    double xPos = uniDist_(gen_) * 20.;
    double yPos = uniDist_(gen_) * 20.;
    double height = heightMap_->getHeight(xPos, yPos);

    gc_init_.head(3) << xPos, yPos, height + 0.2;

    husky_->setState(gc_init_, gv_init_);
    updateObservation();
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    torque4_ = action.cast<double>();
    torque4_ = torque4_.cwiseProduct(actionStd_);
    torque4_ += actionMean_;
    genForce_.tail(nJoints_) = torque4_;
    husky_->setGeneralizedForce(genForce_);

    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++) {
      if(server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if(server_) server_->unlockVisualizationServerMutex();
    }

    updateObservation();
    rewards_.record("goal", gc_.head<2>().norm());
    return rewards_.sum();
  }

  void updateObservation() {
    husky_->getState(gc_, gv_);
    obDouble_ << gc_.head(7), gv_;
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = 0.;
    return false;
  }

  void curriculumUpdate() {

  };

 private:
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* husky_;
  raisim::HeightMap* heightMap_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, genForce_, torque4_;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;

  thread_local static std::mt19937 gen_;
  thread_local static std::normal_distribution<double> normDist_;
  thread_local static std::uniform_real_distribution<double> uniDist_;

};

thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
thread_local std::normal_distribution<double> raisim::ENVIRONMENT::normDist_(0., 1.);
thread_local std::uniform_real_distribution<double> raisim::ENVIRONMENT::uniDist_(-1., 1.);
}

