#pragma once
#include "material.h"

// Light
class light {
public:
    __device__ light(point3 pos, color amb, color dif, color spe) : p(pos), l_ambient(amb), l_diffuse(dif), l_specular(spe) {}

    point3   p;      
    color    l_ambient;   
    color    l_diffuse;
    color    l_specular;
};

class camera {
public:
    __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, int image_width, int image_height) { 
        initialize(lookfrom, lookat, vup, vfov, aspect, image_width, image_height);
    }
    __device__ void initialize(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, int image_width, int image_height);
    __device__ color phong(const vec3& n, const vec3& v, const vec3& l, const vec3& r, material* mat_ptr, const light& light);
   
    __device__ ray get_ray(float i, float j, curandState* local_rand_state);
    __device__ color ray_color(const ray& r, hittable** world, curandState* local_rand_state, int num_hittables);
    

    int     samples_per_pixel = 10;      // Count of random samples for each pixel
    float   focal_length;                // Distance from camera lookfrom point to center of image plane 
    point3  center;                      // Camera center
    point3  pixel00_loc;                 // Location of pixel 0, 0
    vec3    pixel_delta_u;               // Offset to pixel to the right
    vec3    pixel_delta_v;               // Offset to pixel below
    vec3    u, v, w;                     // Camera frame basis vectors
};

__device__ void camera::initialize(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, int image_width, int image_height){
    // Initialize
    center = lookfrom;
    focal_length = (lookfrom - lookat).length();

    // Determine viewport dimensions.
    float theta = degrees_to_radians(vfov);
    float h = tan(theta / 2.0f);
    float viewport_height = 2 * h * focal_length;
    float viewport_width = viewport_height * aspect;

    // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
    w = unit_vector(lookfrom - lookat);
    u = unit_vector(cross(vup, w));
    v = cross(w, u);

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    vec3 viewport_u = viewport_width * u;       // Vector across viewport horizontal edge
    vec3 viewport_v = viewport_height * -v;     // Vector down viewport vertical edge

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    pixel_delta_u = viewport_u / image_width;
    pixel_delta_v = viewport_v / image_height;
    
    // Calculate the location of the upper left pixel.
    vec3 viewport_upper_left = center - (focal_length * w) - viewport_u / 2 - viewport_v / 2;
    pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
}

// Ambient, diffuse, specular
__device__ color camera::phong(const vec3& n, const vec3& v, const vec3& l, const vec3& r, material* mat_ptr, const light& light) {
    color    intensity(0, 0, 0);
    intensity += mat_ptr->getM_Ambient()* light.l_ambient;
    
    float   lambertian = fmaxf(dot(n, l), 0.0f);
    if (lambertian > 0)
    {
        float specular = powf(fmaxf(dot(v, r), 0.0f), mat_ptr -> getM_Shininess());
        intensity += mat_ptr -> getM_Diffuse() * lambertian * light.l_diffuse;
        intensity += mat_ptr -> getM_Specular() * specular * light.l_specular;
    }
    return intensity;
}

// Ramdom ray sampling within its corresponding pixel area
__device__ ray camera::get_ray(float i, float j, curandState* local_rand_state) {
    // Sampling 
    vec3 offset = sample_square(local_rand_state);
    vec3 pixel_sample = pixel00_loc
        + ((i + offset.x()) * pixel_delta_u)
        + ((j + offset.y()) * pixel_delta_v);

    point3 ray_origin = center;
    vec3 ray_direction = unit_vector(pixel_sample - ray_origin);

    return ray(ray_origin, ray_direction);
}

// Compute the intensity from ray using simple ray casting
__device__ color camera::ray_color(const ray& r, hittable** world, curandState* local_rand_state, int num_hittables) {

    ray     primary_ray = r;
    color   intensity(0, 0, 0);

    color   cur_attenuation = color(1.0, 1.0, 1.0);
    hit_record  rec;
    
    // In case which primary ray does not hit anything
    if (!(*world)->hit(primary_ray, interval(0.001f, FLT_MAX), rec)) {
        return color(0, 0, 0);  // Backgorund color
        /*vec3 unit_direction = unit_vector(cur_ray.direction());
        float t = 0.5f * (unit_direction.y() + 1.0f);
        vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
        return cur_attenuation * c;*/
    }
    
    // Light configuration
    light light(point3(0, 100, 0), color(0.2, 0.2, 0.2), color(1.0, 1.0, 1.0), color(1.0, 1.0, 1.0));
    
    // Shadow ray
    hit_record s_rec;
    vec3 shadow_ray_orig = rec.p + rec.normal * 0.01;
    vec3 shadow_ray_dir = unit_vector(light.p - rec.p);
    ray shadow_ray(rec.p, shadow_ray_dir);

    float max_distance = (light.p - shadow_ray.orig).length();
    bool isShadowed = (*world)->hit(shadow_ray, interval(0.01f, max_distance), s_rec);

    if(!isShadowed) {
        vec3 shadow_ray_dir = unit_vector(light.p - rec.p);
        intensity += rec.mat_ptr->getM_Ambient() * light.l_ambient;
        // Parameters for Phong relfection model
        vec3 n = rec.normal;                                            // Normal vector from the Ray-Torus intersection point
        vec3 l = shadow_ray_dir;                                        // Direction to the light source
        vec3 v = unit_vector(center - rec.p);                           // Direction to the viewer
        vec3 refl = unit_vector(reflect(-shadow_ray_dir, n));           // Reflection of light

        intensity += phong(n, v, l, refl, rec.mat_ptr, light);
        return intensity;
    }
        
    intensity += rec.mat_ptr->getM_Ambient() * light.l_ambient;
    return intensity;
}

