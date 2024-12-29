#pragma once
#include "hittable.h"
#include "interval.h"
#include "math.h"
#include "mat4.h"

// Considers a torus by revolving a cirle around the y-axis
class torus : public hittable {
public:
    __device__ torus() {}
    __device__ torus(point3 cen, float major_r, float minor_r, material* m, vec3 rot_ang) : center(cen), R(major_r), r(minor_r), mat_ptr(m), 
                     rotation_angles(vec3(degrees_to_radians(rot_ang.x()), degrees_to_radians(rot_ang.y()), degrees_to_radians(rot_ang.z()) )) {}

    __device__ inline vec3 compute_normal(vec3 p, float R, float r) const;
    __device__ inline vec3 rotate_point(const vec3& p, const vec3& rotation) const;
    __device__ inline vec3 inverse_rotate_point(const vec3& p, const vec3& rotation) const;

    __device__ virtual bool hit(const ray& ray, interval ray_t, hit_record& rec) const;

    point3  center;             // Torus center (Center of hole)
    float R;                    // Major radius (Distance from the center of hole inside torus and the center of its tube)
    float r;                    // Minor radius (Radius of the tube)
    material* mat_ptr;          // Matetrial pointer
    vec3 rotation_angles;       // Rotation angles (in radians)
};

__device__ inline vec3 torus::compute_normal(vec3 p, float R, float r) const{
    float a = 1.0 - (R / sqrtf(p.x() * p.x() + p.z() * p.z()));
    vec3 normal = vec3(a * p.x(), p.y(), a * p.z());

    return normal;
}

__device__ inline vec3 torus::rotate_point(const vec3& p, const vec3& rot_angle) const {
    return rotation_z(rot_angle.z()) * (rotation_y(rot_angle.y()) * (rotation_x(rot_angle.x()) * p));
}

__device__ inline vec3 torus::inverse_rotate_point(const vec3& p, const vec3& rot_angle) const {
    return rotation_x(-rot_angle.x())* (rotation_y(-rot_angle.y()) * (rotation_z(-rot_angle.z()) * p));
}

__device__ bool torus::hit(const ray& ray, interval ray_t, hit_record& rec) const {
    float coeffs[5];
    float roots[4];

    // Inverse transformation for ray
    vec3 ray_orig_torus_space = inverse_rotate_point(ray.orig - center, rotation_angles);            // [TR]^-1 * p
    vec3 ray_dir_torus_space  = inverse_rotate_point(unit_vector(ray.dir) , rotation_angles);        // [TR]^-1 * p
    
    // Set up the coefficients of a quartic equation for the Ray-Torus intersection
    float A = ray_dir_torus_space.length_squared();
    float B = 2.0 * dot(ray_orig_torus_space, ray_dir_torus_space);
    float C = ray_orig_torus_space.length_squared() + R * R - r * r;
    float D = 4.0 * R * R * (ray_dir_torus_space.x()  * ray_dir_torus_space.x()  + ray_dir_torus_space.z()  * ray_dir_torus_space.z() );
    float E = 8.0 * R * R * (ray_orig_torus_space.x() * ray_dir_torus_space.x()  + ray_orig_torus_space.z() * ray_dir_torus_space.z() );
    float F = 4.0 * R * R * (ray_orig_torus_space.x() * ray_orig_torus_space.x() + ray_orig_torus_space.z() * ray_orig_torus_space.z());
   
    coeffs[4] = A * A;
    coeffs[3] = 2.0 * A * B;
    coeffs[2] = 2.0 * A * C + B * B - D;
    coeffs[1] = 2.0 * B * C - E;
    coeffs[0] = C * C - F;  					

    // Find roots of the quartic equation
    
    // Analytical
    //int num_real_roots = solve_quartic(coeffs, roots);
    
    // Numerical
    polynomial<4> poly = { coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4] }; 
    int num_real_roots = poly.find_roots(roots, ray_t.min, ray_t.max, 0.0001f);

    bool	intersected = false;
    float 	t = FLT_MAX;
    
    // Ray misses the torus
    if (num_real_roots == 0) return false; 

    // Find the smallest root greater than 0.001
    for (int j = 0; j < num_real_roots; j++) {
        if (ray_t.surrounds(roots[j])) {
            intersected = true;
            if (roots[j] < t)
                t = roots[j];
        }
    }
    if (!intersected) return false;

    // Record intersection point and its normal vector
    rec.t = t;              
    rec.p = rotate_point(ray_orig_torus_space + t * ray_dir_torus_space, rotation_angles) + center; // A ray hitting a transformed torus in world space

    // Normal vector of the hit point on the transformed torus in world space 
    // (Calculated by transforming the normal vector(R * N) of the corresponding hit point on the untransformed torus in its local(torus) space)
    rec.normal = unit_vector(rotate_point(compute_normal(ray_orig_torus_space + t * ray_dir_torus_space, R, r), rotation_angles));
    rec.mat_ptr = mat_ptr;

    return true;
}