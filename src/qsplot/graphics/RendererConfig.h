#pragma once

/**
 * @brief Configuration for the Renderer.
 * 
 * This struct holds all configurable parameters for the visualization engine.
 * It can be passed to the Renderer constructor or modified at runtime.
 */
struct RendererConfig {
    // Window settings
    int windowWidth = 1280;
    int windowHeight = 720;
    const char* windowTitle = "QsPlot";
    bool vsync = true;
    
    // Visual settings
    float pointScale = 0.05f;
    float globalAlpha = 1.0f;
    int colorMode = 0;  // 0: Heatmap, 1: CoolWarm, 2: Grayscale
    
    // Background color (RGB)
    float backgroundColor[3] = {0.05f, 0.05f, 0.05f};
    
    // Camera settings
    float cameraDistance = 25.0f;
    
    // Default constructor
    RendererConfig() = default;
};
