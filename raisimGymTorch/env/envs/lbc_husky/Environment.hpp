#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"

/// gc_ = x,y,z positions
///       w,x,y,z quaternion
///       w1,w2,w3,w4 wheel angles

/// gv_ = x,y,z linear velocities
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
            //EXP: where you define the robot, definitions of robot is provided in this urdf

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
                    poles_.emplace_back(Eigen::Vector2d{1.01449*j - 35.0, 1.01449*i - 35.0});
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
            // to define configuration robot, position using generalized coordinate gc, velocity using generalized velocity gv
            // to describe states. Generalized coordinate ->  floating base system
            // floating based system: define where the robot is / configuration of robot
            // define 3d position of base (expressed in 3d vector), orientation of base(quaternion), and angle of the four wheels(angles)

            /// this is nominal configuration of anymal
            gc_init_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            // x,y,z, cortonian(4) rotation, 4 wheel angles => 11DOF

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

            // get previous distance
            double prev_distance = gc_.head<2>().norm();
            updateObservation();
            double goal, torque_dir, torque_init, torque_guide, vel_dir, vel_init, torque_end, vel_end, height, progress;
            double cur_distance= gc_.head<2>().norm();
            double good_direction = -5*gv_.head<2>().dot(gc_.head<2>()) / pow(cur_distance,2); // 0.2=0.2 guiding only
            double difference = prev_distance-cur_distance; //0.2 in ave
            if (cur_distance < 2){ // if it has moved very close, increase much more?
                goal = (10+2*(2-cur_distance)); // 10<goal>14*0.1=1.4 // (cur_distance+1e-5)
                rewards_.record("goal", (float) goal);
                vel_end =  0.1*gv_.head<2>().norm(); //5*0.45*0.1=0.225
                torque_end = (torque4_.norm()*0.001); //200*0.001 =0.2
                rewards_.record("velocity", (float) vel_end);
                rewards_.record("torque", (float) torque_end);
                return rewards_.sum();
            }
            else{
                goal = -cur_distance; //goal = -27*0.1s
                if (difference < 0) {difference = difference-1;} //
            }
            progress = 5*difference; // 0.2*5=1
            torque_dir = (torque4_.norm()*0.001*(difference)); // 200*0.001*0.2=0.04
            torque_guide = (torque4_.norm()*0.002*good_direction); //200*0.001*0.1 = 0.02
            vel_dir =  (gv_.head<2>().norm()*(difference)*0.2); // (1.5~2)*0.45*0.2=0.13
            /// initialization
            vel_init =  gv_.head<2>().norm()*(currfac_velocity);
            torque_init = (torque4_.norm()*0.001*(currfac_velocity));//200*0.001
            /// Height
            double z = gc_.segment(2,1).norm();
            if (z> 1.3) {height = (1-currfac_height)*(z-1.3)*-100;}
            // old: at iteration 5000, 0.05*-0.45=0.0225*100=2.25, Now: 0.05*0.2*100=-1
            // if z >1.5, car tries to jump over! Penalize heavily!
//            if (abs(abs(gc_(2)))> 1.5) {rewards_.record("height", (float) (100.0*(abs(gc_(2))-1.8)));}
            //penalize heavily if collision NOT WORKING
//            if (gc_.segment(2,1).norm() > 1.3) {rewards_.record("height", 100.0*(1-currfac_height));}
///             SUM OF EACH COMPONENT
///             NORMALIZE
//            double mean = (goal+good_direction+torque_dir+torque_guide+vel_dir+torque_init+vel_init);
//            double std = sqrt((goal*goal+good_direction*good_direction+torque_dir*torque_dir+torque_guide*torque_guide+vel_dir*vel_dir
//                    + torque_init*torque_init + vel_init*vel_init + 1e-8)/7.0);
//            goal = (goal-mean)/std;
//            good_direction = (good_direction-mean)/std;
//            torque_dir = (torque_dir-mean)/std;
//            torque_guide = (torque_guide-mean)/std;
//            vel_dir = (vel_dir-mean)/std;
//            torque_init = (torque_init-mean)/std;
//            vel_init = (vel_init-mean)/std;

///             SUM UP REWARDS
            rewards_.record("goal", (float) goal);
            rewards_.record("direction", (float) progress);
            rewards_.record("direction", (float) good_direction);
            rewards_.record("torque", (float) torque_dir);
            rewards_.record("torque", (float) torque_guide);
            rewards_.record("torque", (float) torque_init);
            rewards_.record("velocity", (float) vel_init);
            rewards_.record("velocity", (float) vel_dir);
            rewards_.record("height", (float) height);
/// gc_ = x,y,z positions
///       w,x,y,z quaternion
///       w1,w2,w3,w4 wheel angles
// Matrix<T,Dynamic,Dynamic,0,MaxRows,MaxCols> Eigen::EigenBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >
/// gv_ = x,y,z linear velocities
///       w_x,w_y,w_z angular velocities in the world frame
///       s1,s2,s3,s4 wheel speed
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

        void curriculumUpdate() { //IMPLEMENT THIS
//    currfac_distance *= 0.995;
            currfac_height *= 0.99999; currfac_shaky *= 0.9;
            currfac_velocity *= 0.95;
        };

    private:
        int gcDim_, gvDim_, nJoints_;
        bool visualizable_ = false;
        raisim::ArticulatedSystem* husky_;
        raisim::HeightMap* heightMap_;
        Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, genForce_, torque4_;
        Eigen::VectorXd actionMean_, actionStd_, obDouble_;
        std::vector<Eigen::Vector2d> poles_;
        int SCANSIZE = 10;
        int GRIDSIZE = 6;
        std::vector<raisim::Visuals *> scans;  // for visualization
//  double currfac_distance = 1;
        double currfac_velocity = 1;
        double currfac_height =1;
        double currfac_shaky = 1;

        thread_local static std::mt19937 gen_;
        thread_local static std::normal_distribution<double> normDist_;
        thread_local static std::uniform_real_distribution<double> uniDist_;
    };

    thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
    thread_local std::normal_distribution<double> raisim::ENVIRONMENT::normDist_(0., 1.);
    thread_local std::uniform_real_distribution<double> raisim::ENVIRONMENT::uniDist_(-1., 1.);
}

