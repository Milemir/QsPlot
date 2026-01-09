void Renderer::mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    Renderer* self = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    if (!self) return;

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        self->m_mouseLeftDown = (action == GLFW_PRESS);
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        self->m_mouseRightDown = (action == GLFW_PRESS);
    }
}

void Renderer::cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    Renderer* self = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    if (!self || !self->m_camera) return;

    double deltaX = xpos - self->m_lastX;
    double deltaY = ypos - self->m_lastY;

    self->m_lastX = xpos;
    self->m_lastY = ypos;

    // Interaction Logic
    // Left Click: Orbit
    if (self->m_mouseLeftDown) {
        // Sensitivity
        float sens = 0.005f;
        self->m_camera->orbit((float)deltaX * sens, (float)deltaY * sens);
    }
    
    // Right Click: Zoom (alternative to scroll) or Pan
    if (self->m_mouseRightDown) {
        float sens = 0.05f;
        self->m_camera->zoom((float)deltaY * sens);
    }
}

void Renderer::scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    Renderer* self = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    if (!self || !self->m_camera) return;

    float sens = 0.5f;
    self->m_camera->zoom((float)yoffset * sens);
}

void Renderer::framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    Renderer* self = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    if (self && self->m_camera) {
        self->m_camera->setAspect(width, height);
    }
    glViewport(0, 0, width, height);
}

void Renderer::processEvents() {
    // Deprecated in favor of callbacks, but kept if needed for polling logic
}
