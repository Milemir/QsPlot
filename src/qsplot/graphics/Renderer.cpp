#include "Renderer.h"
#include "Shader.h"
#include "Camera.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <cstdio>

// ImGui Headers
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

extern void ImGui_ImplGlfw_MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
extern void ImGui_ImplGlfw_ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);
extern void ImGui_ImplGlfw_CursorPosCallback(GLFWwindow* window, double x, double y);

Renderer::Renderer() 
    : Renderer(RendererConfig{})
{
}

Renderer::Renderer(const RendererConfig& config)
    : m_config(config), m_window(nullptr), m_running(false), m_stagedCount(0), m_renderCount(0), m_forceUpdate(false),
      m_validVAO(0), m_validVBO(0), m_instanceVBO_Pos(0), m_instanceVBO_Val(0), m_shaderProgram(0),
      m_instanceVBO_NextPos(0), m_instanceVBO_NextVal(0), m_stagedNextCount(0), m_forceUpdateNext(false),
      m_camera(nullptr), m_mouseLeftDown(false), m_mouseRightDown(false), m_lastX(0), m_lastY(0),
      m_pointScale(config.pointScale), m_globalAlpha(config.globalAlpha), m_colorMode(config.colorMode), m_morphTime(0.0f),
      m_colorFilterEnabled(false), m_colorFilterValue(0.5f), m_colorFilterTolerance(0.05f),
      m_selectedID(-1), m_pickingFBO(0), m_pickingTexture(0), m_pickingDepth(0), m_pickingShaderProgram(0)
{
}

Renderer::~Renderer() {
    stop();
}

void Renderer::start() {
    if (m_running) return;
    m_running = true;
    m_renderThread = std::thread(&Renderer::loop, this);
}

void Renderer::stop() {
    m_running = false;
    if (m_renderThread.joinable()) {
        m_renderThread.join();
    }
}

void Renderer::setPoints(const float* positions, const float* values, size_t count) {
    std::lock_guard<std::mutex> lock(m_dataMutex);
    
    if (positions && count > 0) {
        m_stagedPositions.assign(positions, positions + count * 3);
    } else {
        m_stagedPositions.clear();
    }
    
    if (values && count > 0) {
        m_stagedValues.assign(values, values + count);
    } else {
        m_stagedValues.clear();
    }
    
    m_stagedCount = count;
    m_forceUpdate = true;
}

void Renderer::setTargetPoints(const float* positions, const float* values, size_t count) {
    std::lock_guard<std::mutex> lock(m_dataMutex);
    
    if (positions && count > 0) {
        m_stagedNextPositions.assign(positions, positions + count * 3);
    } else {
        m_stagedNextPositions.clear();
    }
    
    if (values && count > 0) {
        m_stagedNextValues.assign(values, values + count);
    } else {
        m_stagedNextValues.clear();
    }
    
    m_stagedNextCount = count;
    m_forceUpdateNext = true;
}

void Renderer::setPointsRaw(const float* positions, const float* values, size_t count) {
    setPoints(positions, values, count);
}

void Renderer::setTickers(const std::vector<std::string>& tickers) {
    std::lock_guard<std::mutex> lock(m_dataMutex);
    m_tickers = tickers;
}

std::string Renderer::getSelectedTicker() const {
    if (m_selectedID >= 0 && m_selectedID < (int)m_tickers.size()) {
        return m_tickers[m_selectedID];
    }
    return "";
}

void Renderer::setDimensionLabels(const std::string& colorLabel, 
                                   const std::string& xLabel,
                                   const std::string& yLabel, 
                                   const std::string& zLabel) {
    std::lock_guard<std::mutex> lock(m_dataMutex);
    m_colorLabel = colorLabel;
    m_xLabel = xLabel;
    m_yLabel = yLabel;
    m_zLabel = zLabel;
}

void Renderer::loop() {
    // 1. Init GLFW
    if (!glfwInit()) return;
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Use config for window size and title
    m_window = glfwCreateWindow(m_config.windowWidth, m_config.windowHeight, m_config.windowTitle, NULL, NULL);
    if (!m_window) { glfwTerminate(); return; }
    
    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(m_config.vsync ? 1 : 0); // VSync from config

    // 2. Init GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) return;

    initGL();

    // Setup ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;         

    ImGui::StyleColorsDark();

    // Init ImGui Backend
    ImGui_ImplGlfw_InitForOpenGL(m_window, false);
    ImGui_ImplOpenGL3_Init("#version 410");

    // Setup Camera and Callbacks defined AFTER ImGui init 
    m_camera = new Camera(1280, 720);
    glfwSetWindowUserPointer(m_window, this);
    glfwSetMouseButtonCallback(m_window, mouse_button_callback);
    glfwSetCursorPosCallback(m_window, cursor_position_callback);
    glfwSetScrollCallback(m_window, scroll_callback);
    glfwSetFramebufferSizeCallback(m_window, framebuffer_size_callback);

    // 3. Render Loop
    while (m_running && !glfwWindowShouldClose(m_window)) {
        renderFrame();
        glfwSwapBuffers(m_window);
        processScreenshotRequest();
        glfwPollEvents();
    }
    
    // Cleanup ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    delete m_camera;
    glfwDestroyWindow(m_window);
    glfwTerminate();
}

// Helper functions for CPU-side color calculation (Mirrors Shader Logic)
static void getHeatMapColor(float t, float* r, float* g, float* b) {
    t = std::max(0.0f, std::min(1.0f, t));
    if (t < 0.5f) {
        float f = t * 2.0f;
        // mix(blue, cyan, f)
        *r = 0.0f; 
        *g = f; 
        *b = 1.0f;
    } else {
        float f = (t - 0.5f) * 2.0f;
        // mix(cyan, red, f)
        *r = f;
        *g = 1.0f - f;
        *b = 1.0f - f;
    }
}

static void getCoolWarmColor(float t, float* r, float* g, float* b) {
    t = std::max(0.0f, std::min(1.0f, t));
    if (t < 0.5f) {
        float f = t * 2.0f;
        // mix(vec3(0.2, 0.4, 1.0), vec3(0.9, 0.9, 0.9), f)
        *r = 0.2f + (0.9f - 0.2f) * f;
        *g = 0.4f + (0.9f - 0.4f) * f;
        *b = 1.0f + (0.9f - 1.0f) * f;
    } else {
        float f = (t - 0.5f) * 2.0f;
        // mix(vec3(0.9, 0.9, 0.9), vec3(1.0, 0.2, 0.2), f)
        *r = 0.9f + (1.0f - 0.9f) * f;
        *g = 0.9f + (0.2f - 0.9f) * f;
        *b = 0.9f + (0.2f - 0.9f) * f;
    }
}

void Renderer::renderFrame() {
    // Check for new data
    {
        std::lock_guard<std::mutex> lock(m_dataMutex);
        if (m_forceUpdate && !m_stagedPositions.empty()) {
            glBindBuffer(GL_ARRAY_BUFFER, m_instanceVBO_Pos);
            glBufferData(GL_ARRAY_BUFFER, m_stagedPositions.size() * sizeof(float), m_stagedPositions.data(), GL_DYNAMIC_DRAW);

            glBindBuffer(GL_ARRAY_BUFFER, m_instanceVBO_Val);
            glBufferData(GL_ARRAY_BUFFER, m_stagedValues.size() * sizeof(float), m_stagedValues.data(), GL_DYNAMIC_DRAW);
            
            m_renderCount = m_stagedCount;
            m_forceUpdate = false;
        }
        if (m_forceUpdateNext && !m_stagedNextPositions.empty()) {
            glBindBuffer(GL_ARRAY_BUFFER, m_instanceVBO_NextPos);
            glBufferData(GL_ARRAY_BUFFER, m_stagedNextPositions.size() * sizeof(float), m_stagedNextPositions.data(), GL_DYNAMIC_DRAW);
            glBindBuffer(GL_ARRAY_BUFFER, m_instanceVBO_NextVal);
            glBufferData(GL_ARRAY_BUFFER, m_stagedNextValues.size() * sizeof(float), m_stagedNextValues.data(), GL_DYNAMIC_DRAW);
            m_forceUpdateNext = false;
        }
    }

    // Start ImGui Frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // UI Definition
    {
        ImGui::Begin("Global Controls");
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::Separator();
        
        ImGui::Text("Appearance");
        ImGui::SliderFloat("Point Size", &m_pointScale, 0.01f, 0.2f);
        ImGui::SliderFloat("Alpha", &m_globalAlpha, 0.0f, 1.0f);
        
        const char* colorConfig[] = { "Heatmap (Blue-Red)", "CoolWarm (Div)", "Viridis (Grayscale)" };
        ImGui::Combo("Color Mode", &m_colorMode, colorConfig, IM_ARRAYSIZE(colorConfig));
        
        ImGui::Separator();
        ImGui::Text("Time Series");
        ImGui::SliderFloat("Time Morph", &m_morphTime, 0.0f, 1.0f);

        // Dimension Labels
        ImGui::Separator();
        ImGui::Text("Dimensions");
        ImGui::BulletText("Color: %s", m_colorLabel.c_str());
        
        // Axis labels with colors matching gizmo
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "  X:"); ImGui::SameLine();
        ImGui::Text("%s", m_xLabel.c_str());
        
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "  Y:"); ImGui::SameLine();
        ImGui::Text("%s", m_yLabel.c_str());
        
        ImGui::TextColored(ImVec4(0.3f, 0.3f, 1.0f, 1.0f), "  Z:"); ImGui::SameLine();
        ImGui::Text("%s", m_zLabel.c_str());

        ImGui::Separator();
        ImGui::Text("Selection");
        if (m_selectedID != -1) {
            ImGui::Text("Selected ID: %d", m_selectedID);
            // Show ticker label if available
            std::string ticker = getSelectedTicker();
            if (!ticker.empty()) {
                ImGui::Text("Ticker: %s", ticker.c_str());
            }
            // Show current value
            {
                std::lock_guard<std::mutex> lock(m_dataMutex);
                if (m_selectedID < (int)m_stagedValues.size()) {
                    float val = m_stagedValues[m_selectedID];
                    ImGui::Text("Value: %.4f", val);
                }
            }
        } else {
            ImGui::Text("None");
        }

        // Color Legend
        ImGui::Separator();
        ImGui::Text("Color Legend");
        {
            ImVec2 legendSize(200, 20);
            ImVec2 pos = ImGui::GetCursorScreenPos();
            ImDrawList* drawList = ImGui::GetWindowDrawList();
            
            // Draw gradient bar
            int numSteps = 40;
            float stepWidth = legendSize.x / numSteps;
            for (int i = 0; i < numSteps; i++) {
                float t = (float)i / (numSteps - 1);
                float r, g, b;
                
                if (m_colorMode == 0) { // Heatmap
                    getHeatMapColor(t, &r, &g, &b);
                } else if (m_colorMode == 1) { // CoolWarm
                    getCoolWarmColor(t, &r, &g, &b);
                } else { // Grayscale
                    r = g = b = t;
                }
                
                ImU32 col = IM_COL32((int)(r*255), (int)(g*255), (int)(b*255), 255);
                drawList->AddRectFilled(
                    ImVec2(pos.x + i * stepWidth, pos.y),
                    ImVec2(pos.x + (i + 1) * stepWidth, pos.y + legendSize.y),
                    col
                );
            }
            
            // Draw color filter slider indicator (thin white bar)
            if (m_colorFilterEnabled) {
                float sliderX = pos.x + m_colorFilterValue * legendSize.x;
                drawList->AddLine(
                    ImVec2(sliderX, pos.y - 5),
                    ImVec2(sliderX, pos.y + legendSize.y + 5),
                    IM_COL32(255, 255, 255, 255),
                    3.0f
                );
            }
            
            // Move cursor past the legend
            ImGui::Dummy(legendSize);
            ImGui::Text("0.0                     1.0");
            
            // Color filter slider
            ImGui::Checkbox("Color Filter", &m_colorFilterEnabled);
            if (m_colorFilterEnabled) {
                ImGui::SliderFloat("Filter Value", &m_colorFilterValue, 0.0f, 1.0f);
                ImGui::SliderFloat("Tolerance", &m_colorFilterTolerance, 0.01f, 0.2f, "±%.2f");
            }
        }

    }
    ImGui::End();

    ImGui::Render();

    glClearColor(m_config.backgroundColor[0], m_config.backgroundColor[1], m_config.backgroundColor[2], 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(m_shaderProgram);

    // Update Camera
    if (m_camera) {
        Eigen::Matrix4f view = m_camera->getViewMatrix();
        // Row 0 is Right, Row 1 is Up
        Eigen::Vector3f right = m_camera->getRight();
        Eigen::Vector3f up    = m_camera->getUp();

        glUniform3fv(glGetUniformLocation(m_shaderProgram, "uCameraRight"), 1, right.data());
        glUniform3fv(glGetUniformLocation(m_shaderProgram, "uCameraUp"), 1, up.data());

        Eigen::Matrix4f vp = m_camera->getViewProjectionMatrix();
        glUniformMatrix4fv(glGetUniformLocation(m_shaderProgram, "uVP"), 1, GL_FALSE, vp.data());
        glUniform1f(glGetUniformLocation(m_shaderProgram, "uScale"), m_pointScale); 
        glUniform1f(glGetUniformLocation(m_shaderProgram, "uAlpha"), m_globalAlpha);
        glUniform1i(glGetUniformLocation(m_shaderProgram, "uColorMode"), m_colorMode);
        glUniform1f(glGetUniformLocation(m_shaderProgram, "uTime"), m_morphTime);
        glUniform1i(glGetUniformLocation(m_shaderProgram, "uSelectedID"), m_selectedID);
        glUniform1i(glGetUniformLocation(m_shaderProgram, "uHasSelection"), (m_selectedID != -1));
        
        // Color filter uniforms
        glUniform1i(glGetUniformLocation(m_shaderProgram, "uColorFilterEnabled"), m_colorFilterEnabled);
        glUniform1f(glGetUniformLocation(m_shaderProgram, "uColorFilterValue"), m_colorFilterValue);
        glUniform1f(glGetUniformLocation(m_shaderProgram, "uColorFilterTolerance"), m_colorFilterTolerance);
        
        // Compute Selected Color on CPU to solve shader issues
        float selR = 0.0f, selG = 0.0f, selB = 0.0f;

        {
            std::lock_guard<std::mutex> lock(m_dataMutex);
            if (m_selectedID >= 0 && m_selectedID < (int)m_stagedValues.size()) {
                float v1 = m_stagedValues[m_selectedID];
                // Check bounds for NextValues
                float v2 = (m_selectedID < (int)m_stagedNextValues.size()) ? m_stagedNextValues[m_selectedID] : v1;
                
                // Calculate current interpolated value
                float val = v1 + (v2 - v1) * m_morphTime; 
                
                // Map value to color
                if (m_colorMode == 0) { // Heatmap
                    getHeatMapColor(val, &selR, &selG, &selB);
                } else if (m_colorMode == 1) { // CoolWarm
                    getCoolWarmColor(val, &selR, &selG, &selB);
                } else { // Direct Value (Viridis/Grayscale fallback -> White)
                    float c = std::max(0.0f, std::min(1.0f, val));
                    selR = c; selG = c; selB = c;
                }
            }
        }
        
        glUniform3f(glGetUniformLocation(m_shaderProgram, "uSelectedColor"), selR, selG, selB);
    }

    glBindVertexArray(m_validVAO);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, (GLsizei)m_renderCount);

    // Render Gizmo on top of scene but behind UI
    renderGizmo();

    // Render UI on top
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}



void Renderer::initGL() {
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 

    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    m_shaderProgram = glCreateProgram();
    glAttachShader(m_shaderProgram, vertexShader);
    glAttachShader(m_shaderProgram, fragmentShader);
    glLinkProgram(m_shaderProgram);
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    float quadVertices[] = { -0.5f, 0.5f, 0.0f, -0.5f, -0.5f, 0.0f, 0.5f, 0.5f, 0.0f, 0.5f, -0.5f, 0.0f };

    glGenVertexArrays(1, &m_validVAO);
    glGenBuffers(1, &m_validVBO);

    glBindVertexArray(m_validVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_validVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    // Buffer 1: Current Pos
    glGenBuffers(1, &m_instanceVBO_Pos); 
    glBindBuffer(GL_ARRAY_BUFFER, m_instanceVBO_Pos);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glVertexAttribDivisor(1, 1); 

    // Buffer 2: Current Val
    glGenBuffers(1, &m_instanceVBO_Val); 
    glBindBuffer(GL_ARRAY_BUFFER, m_instanceVBO_Val);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
    glVertexAttribDivisor(2, 1);

    // Buffer 3: Next Pos
    glGenBuffers(1, &m_instanceVBO_NextPos); 
    glBindBuffer(GL_ARRAY_BUFFER, m_instanceVBO_NextPos);
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glVertexAttribDivisor(3, 1); 

    // Buffer 4: Next Val
    glGenBuffers(1, &m_instanceVBO_NextVal); 
    glBindBuffer(GL_ARRAY_BUFFER, m_instanceVBO_NextVal);
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
    glVertexAttribDivisor(4, 1);

    glBindVertexArray(0);

    // Init Picking FBO
    initPickingFBO(1280, 720);

    // Compile Picking Shaders
    unsigned int pickVS = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(pickVS, 1, &pickingVertexShaderSource, NULL);
    glCompileShader(pickVS);
    
    unsigned int pickFS = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(pickFS, 1, &pickingFragmentShaderSource, NULL);
    glCompileShader(pickFS);

    m_pickingShaderProgram = glCreateProgram();
    glAttachShader(m_pickingShaderProgram, pickVS);
    glAttachShader(m_pickingShaderProgram, pickFS);
    glLinkProgram(m_pickingShaderProgram);

    glDeleteShader(pickVS);
    glDeleteShader(pickFS);

    // ---------------------------
    // Gizmo Initialization
    // ---------------------------
    const char* gizmoVert = R"(
        #version 410 core
        layout(location = 0) in vec3 aPos;
        layout(location = 1) in vec3 aColor;
        out vec3 vColor;
        uniform mat4 uVP;
        void main() {
            vColor = aColor;
            gl_Position = uVP * vec4(aPos, 1.0);
        }
    )";
    const char* gizmoFrag = R"(
        #version 410 core
        in vec3 vColor;
        out vec4 FragColor;
        void main() {
            FragColor = vec4(vColor, 1.0);
        }
    )";

    unsigned int gVS = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(gVS, 1, &gizmoVert, NULL);
    glCompileShader(gVS);

    unsigned int gFS = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(gFS, 1, &gizmoFrag, NULL);
    glCompileShader(gFS);

    m_gizmoShaderProgram = glCreateProgram();
    glAttachShader(m_gizmoShaderProgram, gVS);
    glAttachShader(m_gizmoShaderProgram, gFS);
    glLinkProgram(m_gizmoShaderProgram);
    
    glDeleteShader(gVS); 
    glDeleteShader(gFS);

    // Axes Data (X=Red, Y=Green, Z=Blue) Length 10
    float axes[] = {
        // Pos             // Color
        0.0f, 0.0f, 0.0f,  1.0f, 0.0f, 0.0f,
        6.0f, 0.0f, 0.0f,  1.0f, 0.0f, 0.0f,

        0.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f,
        0.0f, 6.0f, 0.0f,  0.0f, 1.0f, 0.0f,

        0.0f, 0.0f, 0.0f,  0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 6.0f,  0.0f, 0.0f, 1.0f
    };

    glGenVertexArrays(1, &m_gizmoVAO);
    glGenBuffers(1, &m_gizmoVBO);
    glBindVertexArray(m_gizmoVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_gizmoVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(axes), axes, GL_STATIC_DRAW);
    
    // Pos
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    // Color
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3*sizeof(float)));

    glBindVertexArray(0);
}
 
void Renderer::processEvents_deprecated() {
    // Deprecated
}

void Renderer::mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    // Forward to ImGui first
    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);

    Renderer* self = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    if (!self) return;

    // Skip if mouse is on ImGui (e.g. menu, slider)
    if (ImGui::GetIO().WantCaptureMouse) return; 

    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        self->m_mouseLeftDown = true;
        
        double x, y;
        glfwGetCursorPos(window, &x, &y);
        
        // Store click start position for click-vs-drag detection
        self->m_clickStartX = x;
        self->m_clickStartY = y;
        self->m_lastX = x;
        self->m_lastY = y;

    } else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
        self->m_mouseLeftDown = false;
        
        double x, y;
        glfwGetCursorPos(window, &x, &y);
        
        // Calculate distance from click start
        double dx = x - self->m_clickStartX;
        double dy = y - self->m_clickStartY;
        double distance = std::sqrt(dx*dx + dy*dy);
        
        // Only select if mouse didn't move much (click, not drag)
        const double CLICK_THRESHOLD = 5.0; // pixels
        if (distance < CLICK_THRESHOLD) {
            // This is a click, do picking
            int picked = self->getPickedID(x, y);
            
            if (picked != -1) {
                self->m_selectedID = picked;
            }
            // If clicked on empty space (picked == -1), preserve current selection
        }
        // If distance >= threshold, it was a drag - don't change selection
        
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        self->m_mouseRightDown = (action == GLFW_PRESS);
    }
}

void Renderer::cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    // Forward to ImGui
    // ImGui internally uses glfwGetCursorPos in NewFrame(), but this helps if needed
    ImGui_ImplGlfw_CursorPosCallback(window, xpos, ypos);

    Renderer* self = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    if (!self || !self->m_camera) return;

    double deltaX = xpos - self->m_lastX;
    double deltaY = ypos - self->m_lastY;

    self->m_lastX = xpos;
    self->m_lastY = ypos;

    if (ImGui::GetIO().WantCaptureMouse) return; 

    // Interaction Logic
    if (self->m_mouseLeftDown) {
        float sens = 0.005f;
        self->m_camera->orbit((float)deltaX * sens, (float)deltaY * sens);
    }
    
    if (self->m_mouseRightDown) {
        float sens = 0.05f;
        // self->m_camera->zoom((float)deltaY * sens); 
        // Right click PAN or ZOOM based on user pref. Let's stick to zoom for now to match prev behavior
        self->m_camera->zoom((float)deltaY * sens);
    }
}

void Renderer::scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    // Forward to ImGui
    ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);

    Renderer* self = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    if (!self || !self->m_camera) return;

    if (ImGui::GetIO().WantCaptureMouse) return;

    float sens = 0.5f;
    self->m_camera->zoom((float)yoffset * sens);
}

void Renderer::framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    Renderer* self = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    if (self && self->m_camera) {
        self->m_camera->setAspect(width, height);
        self->initPickingFBO(width, height); 
    }
    glViewport(0, 0, width, height);
}

void Renderer::renderGizmo() {
    if (m_camera) {
        glUseProgram(m_gizmoShaderProgram);
        Eigen::Matrix4f vp = m_camera->getViewProjectionMatrix();
        glUniformMatrix4fv(glGetUniformLocation(m_gizmoShaderProgram, "uVP"), 1, GL_FALSE, vp.data());
        
        // Draw thicker lines for better visibility
        glLineWidth(2.0f);
        glBindVertexArray(m_gizmoVAO);
        glDrawArrays(GL_LINES, 0, 6);
        glLineWidth(1.0f);
    }
}

void Renderer::saveScreenshot(const std::string& path) {
    std::lock_guard<std::mutex> lock(m_dataMutex);
    m_screenshotPath = path;
    m_screenshotRequested = true;
}

void Renderer::processScreenshotRequest() {
    if (!m_screenshotRequested) return;
    
    int width, height;
    glfwGetFramebufferSize(m_window, &width, &height);
    
    // Allocate buffer for pixel data (RGB)
    std::vector<unsigned char> pixels(width * height * 3);
    
    // Read pixels from framebuffer
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    
    // Flip vertically (OpenGL origin is bottom-left)
    std::vector<unsigned char> flipped(width * height * 3);
    for (int y = 0; y < height; y++) {
        memcpy(
            flipped.data() + y * width * 3,
            pixels.data() + (height - 1 - y) * width * 3,
            width * 3
        );
    }
    
    // Write to file as simple PPM format (portable, no external deps)
    std::string path;
    {
        std::lock_guard<std::mutex> lock(m_dataMutex);
        path = m_screenshotPath;
        m_screenshotRequested = false;
    }
    
    FILE* fp = fopen(path.c_str(), "wb");
    if (fp) {
        fprintf(fp, "P6\n%d %d\n255\n", width, height);
        fwrite(flipped.data(), 1, flipped.size(), fp);
        fclose(fp);
        printf("[Screenshot] Saved to: %s\n", path.c_str());
    } else {
        printf("[Screenshot] ERROR: Could not write to: %s\n", path.c_str());
    }
}
