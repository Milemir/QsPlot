#pragma once

#include <Eigen/Dense>

class Camera {
public:
    Camera(int width, int height);

    void setAspect(int width, int height);
    void update();

    // Controls
    void orbit(float deltaX, float deltaY);
    void zoom(float delta);
    void pan(float deltaX, float deltaY);
    void reset();

    // Getters
    Eigen::Matrix4f getViewMatrix() const;
    Eigen::Matrix4f getProjectionMatrix() const;
    Eigen::Matrix4f getViewProjectionMatrix() const;
    Eigen::Vector3f getPosition() const;
    Eigen::Vector3f getRight() const;
    Eigen::Vector3f getUp() const;

private:
    void updateView();
    void updateProjection();

    // State
    Eigen::Vector3f m_target;
    Eigen::Vector3f m_right;
    Eigen::Vector3f m_up;
    float m_distance;

    float m_yaw;   // Radians
    float m_pitch; // Radians

    // Projection
    float m_fov;
    float m_aspect;
    float m_near;
    float m_far;

    // Matrices
    Eigen::Matrix4f m_view;
    Eigen::Matrix4f m_projection;
};
