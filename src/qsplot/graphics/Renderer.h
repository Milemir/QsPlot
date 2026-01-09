#pragma once

#include <vector>
#include <string>
#include <mutex>
#include <thread>
#include <atomic>

#include "RendererConfig.h"

// Forward declarations to avoid heavy includes in header
struct GLFWwindow;

class Renderer {
public:
    Renderer();
    explicit Renderer(const RendererConfig& config);
    ~Renderer();

    // Start the rendering loop in a separate thread
    void start();

    // Stop the rendering loop
    void stop();

    // positions: N x 3 flattened array
    // values: N x 1 array for color mapping
    void setPoints(const float* positions, const float* values, size_t count);
    
    // Set the target state for morphing
    void setTargetPoints(const float* positions, const float* values, size_t count);

    // Bypasses PCA/Scaling in DataProcessor. Assumes input is already normalized to reasonable range (e.g. -10 to 10)
    void setPointsRaw(const float* positions, const float* values, size_t count);

    // Set ticker labels for each point
    void setTickers(const std::vector<std::string>& tickers);

    // Get selected point's ticker (empty if no selection or no ticker data)
    std::string getSelectedTicker() const;

    // Get selected ID
    int getSelectedID() const { return m_selectedID; }

    // Get/Set configuration
    const RendererConfig& getConfig() const { return m_config; }

    // Request a screenshot to be saved
    void saveScreenshot(const std::string& path);

    // Set dimension labels for UI display
    void setDimensionLabels(const std::string& colorLabel, 
                            const std::string& xLabel,
                            const std::string& yLabel, 
                            const std::string& zLabel);

private:
    void loop(); 
    void initGL();
    void renderFrame();
    void renderGizmo(); // Render Axes

    void processEvents_deprecated();

    // Configuration
    RendererConfig m_config;

    // Interaction with the window
    GLFWwindow* m_window;
    std::atomic<bool> m_running;
    std::thread m_renderThread;

    // Data buffers (Current State) - Now using std::vector for memory safety
    std::mutex m_dataMutex;
    std::vector<float> m_stagedPositions;
    std::vector<float> m_stagedValues;
    size_t m_stagedCount;
    bool m_forceUpdate;

    // Data buffers (Target State) - Now using std::vector for memory safety
    std::vector<float> m_stagedNextPositions;
    std::vector<float> m_stagedNextValues;
    size_t m_stagedNextCount;
    bool m_forceUpdateNext;

    // Values used for CPU-side selection color calculation
    // (Now just references to staged buffers, no separate cache needed)

    // Ticker labels for each point (optional)
    std::vector<std::string> m_tickers;

    // Dimension labels for UI display
    std::string m_colorLabel = "Feature 0";
    std::string m_xLabel = "PCA 1";
    std::string m_yLabel = "PCA 2";
    std::string m_zLabel = "PCA 3";

    // OpenGL Objects
    unsigned int m_validVAO, m_validVBO; 
    unsigned int m_instanceVBO_Pos;     // Current Pos (Loc 1)
    unsigned int m_instanceVBO_Val;     // Current Val (Loc 2)
    unsigned int m_instanceVBO_NextPos; // Target Pos (Loc 3)
    unsigned int m_instanceVBO_NextVal; // Target Val (Loc 4)
    unsigned int m_shaderProgram;

    // Gizmo
    unsigned int m_gizmoVAO, m_gizmoVBO;
    unsigned int m_gizmoShaderProgram;

    size_t m_renderCount;

    // Interaction
    class Camera* m_camera;
    bool m_mouseLeftDown;
    bool m_mouseRightDown;
    double m_lastX, m_lastY;
    double m_clickStartX, m_clickStartY;  // For click-vs-drag detection

    // UI Parameters
    float m_pointScale;
    float m_globalAlpha;
    int m_colorMode; 
    float m_morphTime; 

    // Picking State
    int m_selectedID; // -1 if nothing selected
    unsigned int m_pickingFBO;
    unsigned int m_pickingTexture; // R32I
    unsigned int m_pickingDepth;
    unsigned int m_pickingShaderProgram;

    void initPickingFBO(int width, int height);
    int getPickedID(double mouseX, double mouseY);

    // Screenshot
    std::string m_screenshotPath;
    std::atomic<bool> m_screenshotRequested{false};
    void processScreenshotRequest();

    // GLFW Callbacks
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
    static void framebuffer_size_callback(GLFWwindow* window, int width, int height);
};