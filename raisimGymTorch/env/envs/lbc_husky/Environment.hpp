#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"

/// gc_ = x,y,z positions
///       w,x,y,z quaternion
///       w1,w2,w3,w4 wheel angles

/// gw_ = x,y,z linear velocities
///       w_x,w_y,w_z angular velocities in the world frame
///       s1,s2,s3,s4 wheel speed

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

    /// set the logger for debugging
    raisim::RaiSimMsg::setFatalCallback([](){throw;});

    /// create world
    world_ = std::make_unique<raisim::World>();

    /// add robot
    husky_ = world_->addArticulatedSystem(resourceDir_ + "/husky/husky.urdf");
    husky_->setName("husky");
    husky_->setControlMode(raisim::ControlMode::FORCE_AND_TORQUE);

    /// add heightmap
    raisim::TerrainProperties terrainProperties;
    terrainProperties.frequency = 0.2;
    terrainProperties.zScale = 2.0;
    terrainProperties.xSize = 70.0;
    terrainProperties.ySize = 70.0;
    terrainProperties.xSamples = 70;
    terrainProperties.ySamples = 70;
    terrainProperties.fractalOctaves = 3;
    terrainProperties.fractalLacunarity = 2.0;
    terrainProperties.fractalGain = 0.25;

    std::unique_ptr<raisim::TerrainGenerator> genPtr = std::make_unique<raisim::TerrainGenerator>(terrainProperties);
    std::vector<double> heightVec = genPtr->generatePerlinFractalTerrain();

    /// add obstacles
    for (int i = 0; i < 70; i += GRIDSIZE) {
      for (int j = (i % (GRIDSIZE * GRIDSIZE)) * 2 / GRIDSIZE; j < 70; j += GRIDSIZE) {
        poles_.emplace_back(Eigen::Vector2d{i - 35.0, j - 35.0});
        heightVec[i*70 + j] += 1.;
      }
    }
    heightMap_ = world_->addHeightMap(terrainProperties.xSamples,
                                      terrainProperties.ySamples,
                                      terrainProperties.xSize,
                                      terrainProperties.xSize,
                                      0.,
                                      0.,
                                      heightVec);

    /// get robot data
    gcDim_ = husky_->getGeneralizedCoordinateDim();
    gvDim_ = husky_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_);
    gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_);
    gv_init_.setZero(gvDim_);
    genForce_.setZero(gvDim_);
    torque4_.setZero(nJoints_);

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 17 + SCANSIZE;
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

      /// lidar points visualization
      for(int i = 0; i < SCANSIZE; i++)
        scans.push_back(server_->addVisualBox("box" + std::to_string(i), 0.1, 0.1, 0.1, 1, 0, 0));
    }
  }

  void init() final { }

  void reset() final {
    {
      double xPos, yPos;

      do {
        int i = int((uniDist_(gen_) * .5 + 0.5) * poles_.size());
        xPos = poles_[i](0) + GRIDSIZE / 2.;
        yPos = poles_[i](1) + GRIDSIZE / 2.;
      } while(xPos > 30 || yPos > 30 || xPos < 5 || yPos < 5);

      double height = heightMap_->getHeight(xPos, yPos);
      gc_init_.head(3) << xPos, yPos, height + 0.2;
      husky_->setState(gc_init_, gv_init_);
    }
    updateObservation();
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    torque4_ = action.cast<double>();
    torque4_ = torque4_.cwiseProduct(actionStd_);
    torque4_ += actionMean_;
    genForce_.tail(nJoints_) = torque4_;

    husky_->setGeneralizedForce(genForce_);

    for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
      if (server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if (server_) server_->unlockVisualizationServerMutex();
    }

    updateObservation();
    rewards_.record("goal", gc_.head<2>().norm());
    return rewards_.sum();
  }

  void updateObservation() {
    husky_->getState(gc_, gv_);

    raisim::Vec<3> lidarPos; raisim::Mat<3,3> lidarOri;
    husky_->getFramePosition("imu_joint", lidarPos);
    husky_->getFrameOrientation("imu_joint", lidarOri);

    Eigen::VectorXd lidarData(SCANSIZE);
    Eigen::Vector3d direction;
    const double scanWidth = 2. * M_PI;

    for (int j = 0; j < SCANSIZE; j++) {
      const double yaw = j * M_PI / SCANSIZE * scanWidth - scanWidth * 0.5 * M_PI;
      direction = {cos(yaw), sin(yaw), -0.1 * M_PI};
      direction *= 1. / direction.norm();
      const Eigen::Vector3d rayDirection = lidarOri.e() * direction;
      auto &col = world_->rayTest(lidarPos.e(), rayDirection, 20);
      if (col.size() > 0) {
        lidarData[j] = (col[0].getPosition() - lidarPos.e()).norm();
        if (visualizable_)
          scans[j]->setPosition(col[0].getPosition());
      } else {
        lidarData[j] = 20;
        if (visualizable_)
          scans[j]->setPosition({0,0,100});
      }
    }
    obDouble_ << gc_.head(7), gv_, lidarData;
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    ob = obDouble_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = 0.;
    return false;
  }

  float notCompleted() {
    if (gc_.head(2).norm() < 2)
      return 0.f;
    else
      return 1.f;
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
  std::vector<Eigen::Vector2d> poles_;
  int SCANSIZE = 20;
  int GRIDSIZE = 6;
  std::vector<raisim::Visuals *> scans;  // for visualization

  thread_local static std::mt19937 gen_;
  thread_local static std::normal_distribution<double> normDist_;
  thread_local static std::uniform_real_distribution<double> uniDist_;
};

thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
thread_local std::normal_distribution<double> raisim::ENVIRONMENT::normDist_(0., 1.);
thread_local std::uniform_real_distribution<double> raisim::ENVIRONMENT::uniDist_(-1., 1.);
}

