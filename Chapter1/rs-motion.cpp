// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2019 Intel Corporation. All Rights Reserved.
#include <librealsense2/rs.hpp>
#include <mutex>
#include "example.hpp"          // Include short list of convenience functions for rendering
#include <cstring>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
struct short3
{
    uint16_t x, y, z;
};

#include "d435.h"

void draw_axes()
{
    glLineWidth(2);
    glBegin(GL_LINES);
    // Draw x, y, z axes
    glColor3f(1, 0, 0); glVertex3f(0, 0, 0);  glVertex3f(-1, 0, 0);
    glColor3f(0, 1, 0); glVertex3f(0, 0, 0);  glVertex3f(0, -1, 0);
    glColor3f(0, 0, 1); glVertex3f(0, 0, 0);  glVertex3f(0, 0, 1);
    glEnd();

    glLineWidth(1);
}

void draw_floor()
{
    glBegin(GL_LINES);
    glColor4f(0.4f, 0.4f, 0.4f, 1.f);
    // Render "floor" grid
    for (int i = 0; i <= 8; i++)
    {
        glVertex3i(i - 4, 1, 0);
        glVertex3i(i - 4, 1, 8);
        glVertex3i(-4, 1, i);
        glVertex3i(4, 1, i);
    }
    glEnd();
}

void render_scene(glfw_state app_state)
{
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glColor3f(1.0, 1.0, 1.0);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, 4.0 / 3.0, 1, 40);

    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);

    glLoadIdentity();
    gluLookAt(1, 0, 5, 1, 0, 0, 0, -1, 0);

    glTranslatef(0, 0, +0.5f + app_state.offset_y*0.05f);
    glRotated(app_state.pitch, -1, 0, 0);
    glRotated(app_state.yaw, 0, 1, 0);
    draw_floor();
}

class camera_renderer
{
    std::vector<float3> positions, normals;
    std::vector<short3> indexes;
public:
    // Initialize renderer with data needed to draw the camera
    camera_renderer()
    {
        uncompress_d435_obj(positions, normals, indexes);
    }

    // Takes the calculated angle as input and rotates the 3D camera model accordignly
    void render_camera(float3 theta)
    {

        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE);

        glPushMatrix();
        // Set the rotation, converting theta to degrees
        glRotatef(theta.x * 180 / PI, 0, 0, -1);
        glRotatef(theta.y * 180 / PI, 0, -1, 0);
        glRotatef((theta.z - PI / 2) * 180 / PI, -1, 0, 0);

        draw_axes();

        // Scale camera drawing
        glScalef(0.035, 0.035, 0.035);

        glBegin(GL_TRIANGLES);
        // Draw the camera
        for (auto& i : indexes)
        {
            glVertex3fv(&positions[i.x].x);
            glVertex3fv(&positions[i.y].x);
            glVertex3fv(&positions[i.z].x);
            glColor4f(0.05f, 0.05f, 0.05f, 0.3f);
        }
        glEnd();

        glPopMatrix();

        glDisable(GL_BLEND);
        glFlush();
    }

};

class rotation_estimator
{
    // theta is the angle of camera rotation in x, y and z components
    float3 theta;
    std::mutex theta_mtx;
    float  q0,q1,q2,q3;
    Eigen::Matrix4d P = Eigen::Matrix4d::Identity() ;
    double bias_wx = -0.00075;
    double bias_wy = -0.00258;
    double bias_wz = -0.00095;
    double bias_ax = 0.333852;
    double bias_ay = 0.2032988;
    double bias_az = 0.3043855;
    double noise_gyro = 2.4e-3;        //Gyroscope noise(discrete), rad/s
    double noise_accel = 2.83e-2;       //Accelerometer noise, m/s^2
    /* alpha indicates the part that gyro and accelerometer take in computation of theta; higher alpha gives more weight to gyro, but too high
    values cause drift; lower alpha gives more weight to accelerometer, which is more sensitive to disturbances */
    float alpha = 0.98;
    bool first = true;
    float gravity = 9.81;
    // Keeps the arrival time of previous gyro frame
    double last_ts_gyro = 0;
    
public:
    // Function to calculate the change in angle of motion based on data from gyro
    void process_gyro(rs2_vector gyro_data, double ts)
    {
        if (first) // On the first iteration, use only data from accelerometer to set the camera's initial position
        {
            last_ts_gyro = ts;
            return;
        }
        // Holds the change in angle, as calculated from gyro
        float3 gyro_angle;

        // Initialize gyro_angle with data from gyro
        gyro_angle.x = gyro_data.x; // Pitch
        gyro_angle.y = gyro_data.y; // Yaw
        gyro_angle.z = gyro_data.z; // Roll
        // Compute the difference between arrival times of previous and current gyro frames
        double dt_gyro = (ts - last_ts_gyro) / 1000.0;
        last_ts_gyro = ts;
        // Propagation 
        float3 unbias_gyro;
        unbias_gyro.x = gyro_data.x - bias_wx;
        unbias_gyro.y = gyro_data.y - bias_wy;
        unbias_gyro.z = gyro_data.z - bias_wz;
        // Compute F_k
        Eigen::Matrix4d Omega;
        Omega<<0.0,-unbias_gyro.x,-unbias_gyro.y,-unbias_gyro.z,
        unbias_gyro.x,0.0,unbias_gyro.y,-unbias_gyro.z,
        unbias_gyro.y,-unbias_gyro.z,0.0,unbias_gyro.x,
        unbias_gyro.z,unbias_gyro.y,-unbias_gyro.x,0.0;
        Eigen::Matrix4d F = Eigen::Matrix4d::Identity();
        F = F + 0.5*dt_gyro*Omega;
        // Compute Q_k
        Eigen::MatrixXd G(4,3);
        G<<-q1,-q2,-q3,q0,-q3,q2,q3,q0,-q1,-q2,q1,q0;
        Eigen::Matrix4d Q;
        Q = dt_gyro*dt_gyro*noise_gyro*noise_gyro*(G*G.transpose())/4;
        // Propagate the state and covariance
        std::lock_guard<std::mutex> lock(theta_mtx);

        Eigen::MatrixXd  X(4,1);
        X<<q0,q1,q2,q3;
         X = F * X;
         X = X / X.norm();    
         // uppdate Quaternion
         q0 = X(0,0);
         q1 = X(1,0);
         q2 = X(2,0);
         q3 = X(3,0);
         P = F * P * F.transpose() + Q;
         
        // Change in angle equals gyro measures * time passed since last measurement
        // gyro_angle = gyro_angle * dt_gyro;

        // Apply the calculated change of angle to the current angle (theta)
        
        // theta.add(-gyro_angle.z, -gyro_angle.y, gyro_angle.x);
    }

    void process_accel(rs2_vector accel_data)
    {
        // Holds the angle as calculated from accelerometer data
        float3 accel_angle;
        Eigen::MatrixXd unbias_accel(3,1);
        Eigen::MatrixXd accel_ea(3,1);
        Eigen::MatrixXd accel_predict(3,1);
        Eigen::MatrixXd err_accel(3,1);
        Eigen::MatrixXd H(3,4);
        // move bias
        unbias_accel(0,0) = accel_data.x - bias_ax;
        unbias_accel(1,0) = accel_data.y - bias_ay;
        unbias_accel(2.0) = accel_data.z - bias_az;
        // using unit vector as observation, do normalization
        accel_ea = unbias_accel/unbias_accel.norm();
        accel_predict<<2*(q1*q3-q0*q2),2*(q0*q1+q2*q3),(q0*q0+q3*q3-q1*q1-q2*q2);
        err_accel = accel_ea - accel_predict;
        // Compute the measurement matrix
        H<<-q2,q3,-q0,q1,q1,q0,q3,q2,q0,-q1,-q2,q3;
        H = 2*H;
        //Measurement noise R
        Eigen::Matrix3d R_internal =  (noise_accel / unbias_accel.norm())*(noise_accel / unbias_accel.norm()) * Eigen::Matrix3d::Identity();
        Eigen::Matrix3d R_external = (1-gravity/ unbias_accel.norm())*(1-gravity/ unbias_accel.norm())* Eigen::Matrix3d::Identity();
        Eigen::Matrix3d R = R_internal + R_external;
        // If it is the first iteration, set initial pose of camera according to accelerometer data (note the different handling for Y axis)
        std::lock_guard<std::mutex> lock(theta_mtx);
        // first we need to use the accelerator to initialize the quaternion
        if (first)
        {
            first = false;
         // Calculate rotation angle from accelerometer data
            accel_angle.z = atan2(accel_data.y, accel_data.z);
            accel_angle.x = atan2(accel_data.x, sqrt(accel_data.y * accel_data.y + accel_data.z * accel_data.z));
            theta = accel_angle;
            // Since we can't infer the angle around Y axis using accelerometer data, we'll use PI as a convetion for the initial pose
            theta.y = PI;
            // compute the quaternion
            q0 = cos(theta.x/2)*cos(theta.y/2)*cos(theta.z/2)+sin(theta.x/2)*sin(theta.y/2)*sin(theta.z/2);
            q1 = sin(theta.x/2)*cos(theta.y/2)*cos(theta.z/2)-cos(theta.x/2)*sin(theta.y/2)*sin(theta.z/2);
            q2 = cos(theta.x/2)*sin(theta.y/2)*cos(theta.z/2)+sin(theta.x/2)*cos(theta.y/2)*sin(theta.z/2);
            q3 = cos(theta.x/2)*cos(theta.y/2)*sin(theta.z/2)-sin(theta.x/2)*sin(theta.y/2)*cos(theta.z/2);
            // intialize the covariance as              1e-10*eye(4)
            P = 1e-10*P;
            // P<<1e-10,0.0,0.0,0.0,
            // 0.0,1e-10,0.0,0.0,
            // 0.0,0.0,1e-10,0.0,
            // 0.0,0.0,0.0,1e-10;
            std::cout<<"Initialized Quaternion is: ["<<q0<<","<<q1<<","<<q2<<","<<q3<<"]"<<std::endl;
            std::cout<<"Initialized Covariance Matrix is:"<<std::endl;
            std::cout<<P.matrix()<<std::endl;
        }
        else  // update
        {
            /* 
            Apply Complementary Filter:
                - high-pass filter = theta * alpha:  allows short-duration signals to pass through while filtering out signals
                  that are steady over time, is used to cancel out drift.
                - low-pass filter = accel * (1- alpha): lets through long term changes, filtering out short term fluctuations 
            */
            // theta.x = theta.x * alpha + accel_angle.x * (1 - alpha);
            // theta.z = theta.z * alpha + accel_angle.z * (1 - alpha);
            
            // update and correction
            // Determine K
            Eigen::Matrix3d S = H*P*H.transpose()+R;
            Eigen::MatrixXd K = P*H.transpose()*S.inverse();
            // update X
            Eigen::MatrixXd X(4,1);
            X<<q0,q1,q2,q3;
            X = X + K*err_accel;
            X = X/X.norm();
            q0 = X(0,0);
            q1 = X(1,0);
            q2 = X(2,0);
            q3 = X(3,0);
            // update P
            P = (Eigen::Matrix4d::Identity() - K*H)*P;
            P = (P+P.transpose())/2;
        }
    }
    
    // Returns the current rotation angle
    float3 get_theta()
    {
        std::lock_guard<std::mutex> lock(theta_mtx);
        theta.z = atan2(2*q0*q1+2*q2*q3,1-2*q1*q1+2*q2*q2);
        theta.x = asin(2*q0*q2-2*q1*q3);
        theta.y = atan2(2*(q0*q3+q1*q2),1-2*(q2*q2+q3*q3));
        return theta;
    }
};


bool check_imu_is_supported()
{
    bool found_gyro = false;
    bool found_accel = false;
    rs2::context ctx;
    for (auto dev : ctx.query_devices())
    {
        // The same device should support gyro and accel
        found_gyro = false;
        found_accel = false;
        for (auto sensor : dev.query_sensors())
        {
            for (auto profile : sensor.get_stream_profiles())
            {
                if (profile.stream_type() == RS2_STREAM_GYRO)
                    found_gyro = true;

                if (profile.stream_type() == RS2_STREAM_ACCEL)
                    found_accel = true;
            }
        }
        if (found_gyro && found_accel)
            break;
    }
    return found_gyro && found_accel;
}

int main(int argc, char * argv[]) try
{
    // Before running the example, check that a device supporting IMU is connected
    if (!check_imu_is_supported())
    {
        std::cerr << "Device supporting IMU (D435i) not found";
        return EXIT_FAILURE;
    }

    // Initialize window for rendering
    window app(1280, 720, "RealSense Motion Example");
    // Construct an object to manage view state
    glfw_state app_state(0.0, 0.0);
    // Register callbacks to allow manipulation of the view state
    register_glfw_callbacks(app, app_state);

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    // Create a configuration for configuring the pipeline with a non default profile
    rs2::config cfg;

    // Add streams of gyro and accelerometer to configuration
    cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
    cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);

    // Declare object for rendering camera motion
    camera_renderer camera;
    // Declare object that handles camera pose calculations
    rotation_estimator algo;

    // Start streaming with the given configuration;
    // Note that since we only allow IMU streams, only single frames are produced
    auto profile = pipe.start(cfg, [&](rs2::frame frame)
    {
        // Cast the frame that arrived to motion frame
        auto motion = frame.as<rs2::motion_frame>();
        // If casting succeeded and the arrived frame is from gyro stream
        if (motion && motion.get_profile().stream_type() == RS2_STREAM_GYRO && motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F)
        {
            // Get the timestamp of the current frame
            double ts = motion.get_timestamp();
            // Get gyro measures
            rs2_vector gyro_data = motion.get_motion_data();
            // Call function that computes the angle of motion based on the retrieved measures
            algo.process_gyro(gyro_data, ts);
        }
        // If casting succeeded and the arrived frame is from accelerometer stream
        if (motion && motion.get_profile().stream_type() == RS2_STREAM_ACCEL && motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F)
        {
            // Get accelerometer measures
            rs2_vector accel_data = motion.get_motion_data();
            // Call function that computes the angle of motion based on the retrieved measures
            algo.process_accel(accel_data);
        }
    });

    // Main loop
    while (app)
    {
        // Configure scene, draw floor, handle manipultation by the user etc.
        render_scene(app_state);
        // Draw the camera according to the computed theta
        camera.render_camera(algo.get_theta());
    }
    // Stop the pipeline
    pipe.stop();

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
