#include "Camera.h"
#include <cmath>
#include <algorithm>

// Helper to lookAt manually since we strictly use Eigen
Eigen::Matrix4f lookAt(const Eigen::Vector3f& eye, const Eigen::Vector3f& target, const Eigen::Vector3f& up) {
    Eigen::Vector3f f = (target - eye).normalized();
    Eigen::Vector3f u = up.normalized();
    Eigen::Vector3f s = f.cross(u).normalized();
    Eigen::Vector3f v = s.cross(f);

    Eigen::Matrix4f res;
    res << s.x(), s.y(), s.z(), -s.dot(eye),
           v.x(), v.y(), v.z(), -v.dot(eye),
          -f.x(), -f.y(), -f.z(),  f.dot(eye),
           0,     0,     0,      1;
    return res;
}

Eigen::Matrix4f perspective(float fov, float aspect, float nearP, float farP) {
    float tanHalfFovy = std::tan(fov / 2.0f);
    Eigen::Matrix4f res = Eigen::Matrix4f::Zero();
    res(0, 0) = 1.0f / (aspect * tanHalfFovy);
    res(1, 1) = 1.0f / (tanHalfFovy);
    res(2, 2) = -(farP + nearP) / (farP - nearP);
    res(2, 3) = -(2.0f * farP * nearP) / (farP - nearP);
    res(3, 2) = -1.0f;
    return res;
}

Camera::Camera(int width, int height) 
    : m_target(0, 0, 0), m_distance(5.0f), m_yaw(0.0f), m_pitch(0.0f),
      m_fov(45.0f * 3.14159f / 180.0f), m_near(0.1f), m_far(100.0f) 
{
    setAspect(width, height);
    update();
}

void Camera::setAspect(int width, int height) {
    if (height == 0) height = 1;
    m_aspect = (float)width / (float)height;
    updateProjection();
}

void Camera::update() {
    updateView();
}

void Camera::orbit(float deltaX, float deltaY) {
    m_yaw -= deltaX;
    m_pitch += deltaY;

    // Clamp pitch to avoid gimbal lock
    // Clamp pitch to avoid gimbal lock (Degrees: -89 to +89)
    float limit = 89.0f * 3.14159f / 180.0f;
    m_pitch = std::clamp(m_pitch, -limit, limit);
    
    updateView();
}

void Camera::zoom(float delta) {
    m_distance -= delta * 0.5f;
    if (m_distance < 0.1f) m_distance = 0.1f;
    updateView();
}

void Camera::pan(float deltaX, float deltaY) {
    // Pan logic depends on view vectors
}

Eigen::Matrix4f Camera::getViewMatrix() const { return m_view; }
Eigen::Matrix4f Camera::getProjectionMatrix() const { return m_projection; }
Eigen::Matrix4f Camera::getViewProjectionMatrix() const { return m_projection * m_view; }
Eigen::Vector3f Camera::getPosition() const { 
    // Reconstruct pos from polar
    float camX = m_distance * cos(m_pitch) * sin(m_yaw);
    float camY = m_distance * sin(m_pitch);
    float camZ = m_distance * cos(m_pitch) * cos(m_yaw);
    return m_target + Eigen::Vector3f(camX, camY, camZ);
}

void Camera::updateView() {
    float camX = m_distance * cos(m_pitch) * sin(m_yaw);
    float camY = m_distance * sin(m_pitch);
    float camZ = m_distance * cos(m_pitch) * cos(m_yaw);
    
    Eigen::Vector3f pos = m_target + Eigen::Vector3f(camX, camY, camZ);
    m_view = lookAt(pos, m_target, Eigen::Vector3f(0, 1, 0));

    m_right = m_view.row(0).head<3>();
    m_up    = m_view.row(1).head<3>();
}

Eigen::Vector3f Camera::getRight() const { return m_right; }
Eigen::Vector3f Camera::getUp() const { return m_up; }

void Camera::updateProjection() {
    m_projection = perspective(m_fov, m_aspect, m_near, m_far);
}
