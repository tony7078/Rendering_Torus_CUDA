#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "mat4.h"
#include "ray.h"
#include "sphere.h"
#include "math.h"
#include "torus.h"
#include "hittable_list.h"
#include "camera.h"
#include "material.h"
#include "interval.h"
#include <device_launch_parameters.h>
#include <time.h>

#define RND (curand_uniform(&local_rand_state))                                 // Range [0,1)
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )      // In order to check errors from CUDA API call results

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        
        // CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Initialize cuRAND state for a single thread
__global__ void rand_init(curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);            // Custom seed : 1984
    }
}

// Initialize cuRAND state for each pixel, ensuring a unique random sequence for each pixel
__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;      // range [0, 1200]
    int j = threadIdx.y + blockIdx.y * blockDim.y;      // range [0, 679]

    if ((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j * max_x + i;

    // Each thread gets different seed number (Each ramdom number generation pattern must be independent per each thread)
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);   
}

// Calculate multi-sampled pixel-based color and storage in the frame buffer
__global__ void render(vec3* fb, int max_x, int max_y, int samples_per_pixel, camera** d_camera, hittable** d_world, curandState* rand_state, int num_hittables) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;      // Range [0, 1200]
    int j = threadIdx.y + blockIdx.y * blockDim.y;      // Range [0, 679]

    if ((i >= max_x) || (j >= max_y)) return;           // Maximun range check according to the screen resolution

    // 2D image-to-1D image transform
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];

    color col(0, 0, 0);
    for (int s = 0; s < samples_per_pixel; s++) {
        ray r = (*d_camera)->get_ray(i, j, &local_rand_state);
        col  += (*d_camera)->ray_color(r, d_world, &local_rand_state, num_hittables);
    }

    rand_state[pixel_index] = local_rand_state;

    // Calculate avg value of pixel
    col /= float(samples_per_pixel);         

    // Gamma transform
    col[0] = linear_to_gamma(col[0]);       
    col[1] = linear_to_gamma(col[1]);     
    col[2] = linear_to_gamma(col[2]);  

    fb[pixel_index] = col;
}

__global__ void create_world(hittable** objects, hittable** d_world, camera** d_camera, int image_width, int image_height, curandState* rand_state, int num_hittables) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;

        // Generate random torus objects
        for (int i = 0; i < num_hittables; i++) {
            // Random position
            float x = RND * 80.0f - 40.0f;  
            float y = RND * 50.0f - 25.0f;
            float z = RND * 90.0f;

            // Random radii
            float major_r = RND * 0.25f + 0.2f;  // 0.6 to 1.3
            float minor_r = RND * 0.15f + 0.1f;  // 0.2 to 0.7

            // Random rotation angles
            float rot_x = RND * 360.0f;
            float rot_y = RND * 360.0f;
            float rot_z = RND * 360.0f;

            // Random color for material
            color ambient = color(RND, RND, RND);
            color diffuse = color(RND, RND, RND);
            color specular = color(0.9f, 0.9f, 0.9f);

            // Create material
            material* mat_ptr = new material(ambient, diffuse, specular, 10.0f);

            // Add torus to objects
            objects[i] = new torus(vec3(x, y, z), major_r, minor_r, mat_ptr, vec3(rot_x, rot_y, rot_z));
        }

        // Ring Torus Setup
        //objects[0] = new torus(vec3(0, 0, 2.0), 4.0, 0.8, new material(color(0.1, 0.1, 0.1), color(0, 0.3, 0.5), color(0.9, 0.9, 0.9), 10.0f), vec3(0, 0, 0));
        //objects[1] = new torus(vec3(0, 0, -2.0), 4.0, 0.8, new material(color(0.1, 0.1, 0.1), color(0, 0.3, 0.5), color(0.9, 0.9, 0.9), 10.0f), vec3(0, 0, 90));

        *rand_state = local_rand_state;
        *d_world = new hittable_list(objects, num_hittables);
        
        // Camera Setup
        
        // Side view (COP along x-axis)
        //vec3 lookfrom(8, 0.5, 0);
        //vec3 lookat(7, 0.5, 0);
        
        // Top view
        //vec3 lookfrom(0, 10, 0);
        //vec3 lookat(0, 9, 0);
        //vec3 vup(0, 0, -1);

        // One Weekend view
        //vec3 lookfrom(10, 8, 0);
        //vec3 lookat(0, 0, -1);

        // Far view
        vec3 lookfrom(0, 0, 100.0f);
        vec3 lookat(0, 0, 90.0f);

        vec3 vup(0, 1, 0);
        float vfov = 90.0f;
        float aspect = float(image_width) / float(image_height);

        *d_camera = new camera(lookfrom, lookat, vup, vfov, aspect, image_width, image_height);
    }
}

__global__ void free_world(hittable** d_object_list, hittable** d_world, camera** d_camera, int num_hittables) {
    delete ((sphere*)d_object_list[0])->mat_ptr;
    for (int i = 1; i < num_hittables; i++) {
        delete ((torus*)d_object_list[i])->mat_ptr;
        delete d_object_list[i];
    }
    delete* d_world;
    delete* d_camera;
}

int main() {
    // Resolution Setup
    int image_width = 1200; 
    int image_height = 675;
    int sample_per_pixel = 100;                  // Sample number of each pixel.
    int tx = 8;                                  // Thread x dimension
    int ty = 8;                                  // Thread y dimension

    int num_pixels = image_width * image_height;
    size_t fb_size = num_pixels * sizeof(vec3);

    // Allocate Frame Buffer 1 (in GPU memory)
    vec3* frame_buffer_1;
    checkCudaErrors(cudaMalloc((void**)&frame_buffer_1, fb_size));

    // Allocate random states
    curandState* d_rand_state_1;                // For rendering        
    checkCudaErrors(cudaMalloc((void**)&d_rand_state_1, num_pixels * sizeof(curandState)));
    curandState* d_rand_state_2;                // For world creation
    checkCudaErrors(cudaMalloc((void**)&d_rand_state_2, 1 * sizeof(curandState)));

    rand_init << <1, 1 >> > (d_rand_state_2);   // 2nd random state initialization
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Scene Setup
    hittable** d_object_list;
    int num_hittables = 10000;                      // Number of total objects
    checkCudaErrors(cudaMalloc((void**)&d_object_list, num_hittables * sizeof(hittable*)));

    hittable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));

    camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));

    create_world << <1, 1 >> > (d_object_list, d_world, d_camera, image_width, image_height, d_rand_state_2, num_hittables);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Performance measurement Setup
    clock_t start, stop;
    start = clock();

    // Render our buffer
    dim3 blocks(image_width / tx + 1, image_height / ty + 1);
    dim3 threads(tx, ty);

    render_init << <blocks, threads >> > (image_width, image_height, d_rand_state_1);   // 1st random state initialization
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render << <blocks, threads >> > (frame_buffer_1, image_width, image_height, sample_per_pixel, d_camera, d_world, d_rand_state_1, num_hittables);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Allocate Frame Buffer 2 (in CPU memory)
    vec3* frame_buffer_2 = (vec3*)malloc(fb_size);
    checkCudaErrors(cudaMemcpy(frame_buffer_2, frame_buffer_1, fb_size, cudaMemcpyDeviceToHost));

    // Check time
    stop = clock();
    float timer_seconds = ((float)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Took " << timer_seconds << " seconds.\n";


    // Directly making a PPM image
    FILE* f = fopen("image.ppm", "w");
    std::fprintf(f, "P3\n%d %d\n%d\n", image_width, image_height, 255);
    static const interval intensity(0.000, 0.999);
    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            size_t pixel_index = j * image_width + i;
            auto pixel_color = frame_buffer_2[pixel_index];
            auto r = pixel_color.x();
            auto g = pixel_color.y();
            auto b = pixel_color.z();

            int rbyte = int(256 * intensity.clamp(r));
            int gbyte = int(256 * intensity.clamp(g));
            int bbyte = int(256 * intensity.clamp(b));

            std::fprintf(f, "%d %d %d ", int(rbyte), int(gbyte), int(bbyte));
        }
    }
    std::clog << "\rDone.                 \n";

    // Freedom for all resources
    checkCudaErrors(cudaDeviceSynchronize());
    free_world << <1, 1 >> > (d_object_list, d_world, d_camera, num_hittables);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_object_list));
    checkCudaErrors(cudaFree(d_rand_state_1));
    checkCudaErrors(cudaFree(d_rand_state_2));
    checkCudaErrors(cudaFree(frame_buffer_1));
    free(frame_buffer_2);

    cudaDeviceReset();
}