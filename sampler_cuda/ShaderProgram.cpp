// Copyright 2004-present Facebook. All Rights Reserved.

#include <pangolin/gl/glsl.h>

constexpr const char* shaderText = R"Shader(
@start vertex
#version 330 core

layout(location = 0) in vec3 vertex;

out vec4 position_world;
out vec4 position_camera;
out vec3 viewDirection_camera;

uniform mat4 MVP;
uniform mat4 V;

void main(){

    // Projected image coordinate
    gl_Position =  MVP * vec4(vertex,1);

    // world coordinate location of the vertex
    position_world = vec4(vertex,1);
    position_camera = V * vec4(vertex, 1);

    viewDirection_camera = normalize(vec3(0,0,0) - position_camera.xyz);

}

@start geometry
#version 330

layout ( triangles ) in;
layout ( triangle_strip, max_vertices = 3 ) out;

in vec4 position_world[];
in vec3 viewDirection_camera[];

out vec3 normal_camera;
out vec3 normal_world;
out vec4 xyz_world;
out vec3 viewDirection_cam;
out vec4 xyz_camera;

uniform mat4 V;

void main()
{
    vec3 A = position_world[1].xyz - position_world[0].xyz;
    vec3 B = position_world[2].xyz - position_world[0].xyz;
    vec3 normal = normalize(cross(A,B));
    vec3 normal_cam = (V * vec4(normal,0)).xyz;

    gl_Position = gl_in[0].gl_Position;
    normal_camera = normal_cam;
    normal_world = normal;
    xyz_world = position_world[0];
    xyz_camera = V * xyz_world;
    viewDirection_cam = viewDirection_camera[0];
    gl_PrimitiveID = gl_PrimitiveIDIn;
    EmitVertex();

    gl_Position = gl_in[1].gl_Position;
    normal_camera = normal_cam;
    normal_world = normal;
    xyz_world = position_world[1];
    xyz_camera = V * xyz_world;
    viewDirection_cam = viewDirection_camera[1];
    gl_PrimitiveID = gl_PrimitiveIDIn;

    EmitVertex();

    gl_Position = gl_in[2].gl_Position;
    normal_camera = normal_cam;
    normal_world = normal;
    xyz_world = position_world[2];
    xyz_camera = V * xyz_world;
    viewDirection_cam = viewDirection_camera[2];
    gl_PrimitiveID = gl_PrimitiveIDIn;

    EmitVertex();
    EndPrimitive();
}

@start fragment
#version 330 core

in vec3 viewDirection_cam;
in vec3 normal_world;
in vec3 normal_camera;
in vec4 xyz_world;
in vec4 xyz_camera;
in int gl_PrimitiveID ;

layout(location = 0) out vec4 out_xyz;
layout(location = 1) out vec4 out_normal;

void main(){
    vec3 view_vector = vec3(0,0,1);

    // Check if we need to flip the normal.
    vec3 normal_world_cor;
    float d = dot(normalize(normal_camera), normalize(view_vector));

    if (abs(d) < 0.001) {
        out_xyz = vec4(0,0,0,0);
        out_normal = vec4(0,0,0,0);
        return;
    }
    else{
        if (d < 0) {
            normal_world_cor = -normal_world;
        } else {
            normal_world_cor= normal_world;
        }

        out_xyz = xyz_world;
        out_xyz.w = gl_PrimitiveID + 1.0f;

        out_normal = vec4(normalize(normal_world_cor),1);
        out_normal.w = gl_PrimitiveID + 1.0f;
    }
}
)Shader";

pangolin::GlSlProgram GetShaderProgram() {
  pangolin::GlSlProgram program;

  program.AddShader(pangolin::GlSlAnnotatedShader, shaderText);
  program.Link();

  return program;
}
