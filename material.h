#pragma once

class hit_record;

class material {
public:
    __device__ material(const color& m_amb, const color& m_dif, const color& m_spec, float m_shi) 
        : m_ambient(m_amb), m_diffuse(m_dif), m_specular(m_spec), m_shininess(m_shi) {}

    __device__ color getM_Ambient()     const { return m_ambient; }
    __device__ color getM_Diffuse()     const { return m_diffuse; }
    __device__ color getM_Specular()    const { return m_specular; }
    __device__ float getM_Shininess()   const { return m_shininess; }

private:
    color   m_ambient;
    color   m_diffuse;
    color   m_specular;
    float   m_shininess;
};