#pragma once

class mat4 {
public:
    __host__ __device__ mat4() {}
    __host__ __device__ mat4(float a00, float a01, float a02, float a03, float a10, float a11, float a12, float a13,
        float a20, float a21, float a22, float a23, float a30, float a31, float a32, float a33) {
        m[0][0] = a00; m[0][1] = a01; m[0][2] = a02; m[0][3] = a03;
        m[1][0] = a10; m[1][1] = a11; m[1][2] = a12; m[1][3] = a13;
        m[2][0] = a20; m[2][1] = a21; m[2][2] = a22; m[2][3] = a23;
        m[3][0] = a30; m[3][1] = a31; m[3][2] = a32; m[3][3] = a33;
    }

    // Multiply mat4 by vec3
    __host__ __device__ inline vec3 operator*(const vec3& v) const {
        float x = m[0][0] * v.x() + m[0][1] * v.y() + m[0][2] * v.z() + m[0][3];
        float y = m[1][0] * v.x() + m[1][1] * v.y() + m[1][2] * v.z() + m[1][3];
        float z = m[2][0] * v.x() + m[2][1] * v.y() + m[2][2] * v.z() + m[2][3];
        float w = m[3][0] * v.x() + m[3][1] * v.y() + m[3][2] * v.z() + m[3][3];

        return vec3(x / w, y / w, z / w); // Homogeneous division
    }

    float m[4][4];
};

__host__ __device__ inline mat4 unit_matrix() {
    return mat4(1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1);
}

// Create a rotation matrix for X, Y, Z axes
__host__ __device__ inline mat4 rotation_x(float angle) {
    if (angle == 0) return unit_matrix();

    float c = cos(angle), s = sin(angle);
    return mat4(1, 0,  0, 0,
                0, c, -s, 0,
                0, s,  c, 0,
                0, 0,  0, 1);
}

__host__ __device__ inline mat4 rotation_y(float angle) {
    if (angle == 0) return unit_matrix();

    float c = cos(angle), s = sin(angle);
    return mat4( c, 0, s, 0,
                 0, 1, 0, 0,
                -s, 0, c, 0,
                 0, 0, 0, 1);
}

__host__ __device__ inline mat4 rotation_z(float angle) {
    if (angle == 0) return unit_matrix();

    float c = cos(angle), s = sin(angle);
    return mat4(c, -s, 0, 0,
                s,  c, 0, 0,
                0,  0, 1, 0,
                0,  0, 0, 1);
}

// Create translation matrix
__host__ __device__ inline mat4 translation(float tx, float ty, float tz) {
    return mat4(1, 0, 0, tx,
                0, 1, 0, ty,
                0, 0, 1, tz,
                0, 0, 0, 1);
}

// Create scaling matrix
__host__ __device__ inline mat4 scaling(float sx, float sy, float sz) {
    return mat4(sx,  0,  0, 0,
                 0, sy,  0, 0,
                 0,  0, sz, 0,
                 0,  0,  0, 1);
}
