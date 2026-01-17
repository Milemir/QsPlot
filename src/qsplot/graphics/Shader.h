#ifndef SHADER_H
#define SHADER_H

const char* vertexShaderSource = R"(
    #version 410 core
    layout(location = 0) in vec3 aLocalPos;    
    layout(location = 1) in vec3 aInstancePos; 
    layout(location = 2) in float aValue;      
    layout(location = 3) in vec3 aNextPos;     
    layout(location = 4) in float aNextValue;  

    out float vValue;
    out vec2 vUV;
    flat out int vID; 

    uniform mat4 uVP;       
    uniform float uScale;   
    uniform float uTime;    
    uniform vec3 uCameraRight;
    uniform vec3 uCameraUp;

    void main() {
        
        vec3 currentPos = mix(aInstancePos, aNextPos, uTime);
        float currentValue = mix(aValue, aNextValue, uTime);

        vValue = currentValue;
        vUV = aLocalPos.xy * 2.0; 
        vID = gl_InstanceID;
        
        vec3 offset = (uCameraRight * aLocalPos.x * uScale) + (uCameraUp * aLocalPos.y * uScale);
        vec3 worldPos = currentPos + offset;
        vec4 clipPos = uVP * vec4(worldPos, 1.0);        
        
        if (clipPos.w < 0.01) {
            
            clipPos = vec4(10.0, 10.0, 10.0, 1.0);
        }
        
        gl_Position = clipPos;
    }
)";

const char* fragmentShaderSource = R"(
    #version 410 core
    in float vValue;
    in vec2 vUV;
    flat in int vID;

    out vec4 FragColor;
    
    uniform float uAlpha;         
    uniform int uColorMode;       
    uniform int uSelectedID;      
    uniform bool uHasSelection;   
    uniform vec3 uSelectedColor;
    
    // Color filter uniforms
    uniform bool uColorFilterEnabled;
    uniform float uColorFilterValue;
    uniform float uColorFilterTolerance;

    
    vec3 heatMap(float t) {
        t = clamp(t, 0.0, 1.0);
        vec3 color = vec3(0.0);
        if (t < 0.5) {
            color = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), t * 2.0);
        } else {
            color = mix(vec3(0.0, 1.0, 1.0), vec3(1.0, 0.0, 0.0), (t - 0.5) * 2.0);
        }
        return color;
    }

    vec3 coolWarm(float t) {
        t = clamp(t, 0.0, 1.0);
        if (t < 0.5) {
            return mix(vec3(0.2, 0.4, 1.0), vec3(0.9, 0.9, 0.9), t * 2.0);
        } else {
            return mix(vec3(0.9, 0.9, 0.9), vec3(1.0, 0.2, 0.2), (t - 0.5) * 2.0);
        }
    }

    void main() {
        
        float distSq = dot(vUV, vUV);
        if (distSq > 1.0) discard;

        // Calculate inner color
        vec3 cv;
        if (uColorMode == 0) cv = heatMap(vValue);
        else if (uColorMode == 1) cv = coolWarm(vValue);
        else cv = vec3(clamp(vValue, 0.0, 1.0));

        // Anti-aliased Outline: Smoothly blend to black at the edge
        // Transition starts at radius 0.92, fully black by 0.95
        float dist = sqrt(distSq);
        float outlineFactor = smoothstep(0.92, 0.95, dist);
        cv = mix(cv, vec3(0.0), outlineFactor);

        float baseAlpha = 1.0; 
        float finalAlpha = baseAlpha * uAlpha;

        if (uHasSelection) {
            if (vID == uSelectedID) {
                finalAlpha = baseAlpha * 1.0; 
                cv = vec3(1.0); 
            } else {
                
                float colorDist = distance(cv, uSelectedColor);
                
                float similarity = 1.0 - clamp(colorDist / 1.5, 0.0, 1.0); 
                
                float lowAlpha = uAlpha;
                float highAlpha = max(uAlpha, 0.3); 

                float dynamicAlpha = mix(lowAlpha, highAlpha, pow(similarity, 3.0));
                
                finalAlpha = baseAlpha * dynamicAlpha;
            }
        }

        if (uColorFilterEnabled) {
            float valueDiff = abs(vValue - uColorFilterValue);
            if (valueDiff > uColorFilterTolerance) {
                discard;
            }
        }

        FragColor = vec4(cv, finalAlpha);
    }
)";





const char* pickingVertexShaderSource = R"(
    #version 410 core
    layout(location = 0) in vec3 aLocalPos;    
    layout(location = 1) in vec3 aInstancePos; 
    layout(location = 2) in float aValue;      
    layout(location = 3) in vec3 aNextPos;     
    layout(location = 4) in float aNextValue;  

    uniform mat4 uVP;       
    uniform float uScale;   
    uniform float uTime;    
    uniform vec3 uCameraRight;
    uniform vec3 uCameraUp;

    
    flat out int vID; 
    out vec2 vUV; 

    void main() {
        vec3 currentPos = mix(aInstancePos, aNextPos, uTime);
        vUV = aLocalPos.xy * 2.0; 
        
        vec3 offset = (uCameraRight * aLocalPos.x * uScale) + (uCameraUp * aLocalPos.y * uScale);
        vec3 worldPos = currentPos + offset;

        vec4 clipPos = uVP * vec4(worldPos, 1.0);
        
        
        if (clipPos.w < 0.01) {
            clipPos = vec4(10.0, 10.0, 10.0, 1.0);
        }
        
        gl_Position = clipPos;
        
        
        vID = gl_InstanceID;
    }
)";

const char* pickingFragmentShaderSource = R"(
    #version 410 core
    layout(location = 0) out int FragID; 

    flat in int vID;
    in vec2 vUV; 

    void main() {
        
        if (dot(vUV, vUV) > 1.0) discard; 
        
        
        FragID = vID; 
    }
)";

#endif