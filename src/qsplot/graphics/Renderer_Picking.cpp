#include "Renderer.h"
#include "Camera.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <Eigen/Dense>

void Renderer::initPickingFBO(int width, int height) {
    if (width <= 0 || height <= 0) return;

    // Delete existing resources if resizing
    if (m_pickingFBO) {
        glDeleteFramebuffers(1, &m_pickingFBO);
        glDeleteTextures(1, &m_pickingTexture);
        glDeleteRenderbuffers(1, &m_pickingDepth);
    }

    glGenFramebuffers(1, &m_pickingFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, m_pickingFBO);

    // 1. Texture Attachment (Integer format for IDs)
    glGenTextures(1, &m_pickingTexture);
    glBindTexture(GL_TEXTURE_2D, m_pickingTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I, width, height, 0, GL_RED_INTEGER, GL_INT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_pickingTexture, 0);

    // 2. Depth Attachment (Renderbuffer) - CRITICAL for depth testing
    glGenRenderbuffers(1, &m_pickingDepth);
    glBindRenderbuffer(GL_RENDERBUFFER, m_pickingDepth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_pickingDepth);

    // 3. Verify
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cout << "ERROR::FRAMEBUFFER:: Picking FBO is not complete!" << std::endl;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

int Renderer::getPickedID(double mouseX, double mouseY) {
    if (!m_pickingFBO || !m_camera) return -1;

    // 1. Sace current OpenGL settings
    GLint lastViewport[4]; glGetIntegerv(GL_VIEWPORT, lastViewport);
    GLboolean lastScissor = glIsEnabled(GL_SCISSOR_TEST);
    GLboolean lastBlend = glIsEnabled(GL_BLEND);
    GLboolean lastDither = glIsEnabled(GL_DITHER);

    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_DITHER);
    glEnable(GL_DEPTH_TEST);

    int fbWidth, fbHeight;
    glfwGetFramebufferSize(m_window, &fbWidth, &fbHeight);
    int winWidth, winHeight;
    glfwGetWindowSize(m_window, &winWidth, &winHeight);

    if (winWidth == 0 || winHeight == 0) return -1;

    glBindFramebuffer(GL_FRAMEBUFFER, m_pickingFBO);
    glViewport(0, 0, fbWidth, fbHeight);
    
    int clearVal = -1;
    glClearBufferiv(GL_COLOR, 0, &clearVal);
    glClear(GL_DEPTH_BUFFER_BIT);

    glUseProgram(m_pickingShaderProgram);

    Eigen::Matrix4f vp = m_camera->getViewProjectionMatrix();
    glUniformMatrix4fv(glGetUniformLocation(m_pickingShaderProgram, "uVP"), 1, GL_FALSE, vp.data());
    glUniform1f(glGetUniformLocation(m_pickingShaderProgram, "uScale"), m_pointScale); 
    glUniform1f(glGetUniformLocation(m_pickingShaderProgram, "uTime"), m_morphTime);
    
    Eigen::Vector3f right = m_camera->getRight();
    Eigen::Vector3f up    = m_camera->getUp();
    glUniform3fv(glGetUniformLocation(m_pickingShaderProgram, "uCameraRight"), 1, right.data());
    glUniform3fv(glGetUniformLocation(m_pickingShaderProgram, "uCameraUp"), 1, up.data());

    glBindVertexArray(m_validVAO);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, (GLsizei)m_renderCount);

    glFlush();
    glFinish();

    double scaleX = (double)fbWidth / (double)winWidth;
    double scaleY = (double)fbHeight / (double)winHeight;
    double invertedY = (double)winHeight - mouseY; // Y Ekseni Tersi

    int readX = (int)(mouseX * scaleX);
    int readY = (int)(invertedY * scaleY);

    int id = -1;
    if (readX >= 0 && readX < fbWidth && readY >= 0 && readY < fbHeight) {
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
             std::cerr << "[Picking] FBO Incomplete! Aborting read." << std::endl;
             return -1;
        }

        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadBuffer(GL_COLOR_ATTACHMENT0); 
        
        glReadPixels(readX, readY, 1, 1, GL_RED_INTEGER, GL_INT, &id);
        
        GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
             std::cerr << "[Picking] glReadPixels Error: " << err << std::endl;
             id = -1; 
        }

        glPixelStorei(GL_PACK_ALIGNMENT, 4);
    } else {
        id = -1;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(lastViewport[0], lastViewport[1], lastViewport[2], lastViewport[3]);
    if (lastScissor) glEnable(GL_SCISSOR_TEST);
    if (lastBlend) glEnable(GL_BLEND);
    if (lastDither) glEnable(GL_DITHER);

    return id;
}
