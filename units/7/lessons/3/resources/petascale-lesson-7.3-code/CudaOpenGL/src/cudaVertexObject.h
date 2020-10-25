#define CUDAVERTEXBUFFEROBJECT_H
#define CUDAVERTEXBUFFEROBJECT_H

// ==================== Libraries ==================
// Depending on the operating system we use
// The paths to SDL are actually different.
// The #define statement should be passed in
// when compiling using the -D argument.
// This gives an example of how a programmer
// may support multiple platforms with different
// dependencies.
#if defined(LINUX) || defined(MINGW)
	#include <SDL2/SDL.h>
#else
	// Windows and Mac use a different path
	// If you have compilation errors, change this as needed.
	#include <SDL.h>
#endif

class CudaEnabledVertexBufferObject{
    public:
        void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
                       unsigned int vbo_res_flags)
        {
            // create buffer object
            glGenBuffers(1, vbo);
            glBindBuffer(GL_ARRAY_BUFFER, *vbo);
            // initialize buffer object
            unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
            glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            // register this buffer object with CUDA
            cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags);
        }
        
        // Delete VBO
        void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
        {
            // unregister this buffer object with CUDA
            cudaGraphicsUnregisterResource(vbo_res);
            glBindBuffer(1, *vbo);
            glDeleteBuffers(1, vbo);
            *vbo = 0;
        }

        void runCuda(struct cudaGraphicsResource **vbo_resource)
        {
            // map OpenGL buffer object for writing from CUDA
            float4 *dptr;
            cudaGraphicsMapResources(1, vbo_resource, 0);
            size_t num_bytes;
            cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
                                                                 *vbo_resource);
            launch_kernel(dptr, mesh_width, mesh_height, g_fAnim);
            // unmap buffer object
            cudaGraphicsUnmapResources(1, vbo_resource, 0);
        }
    private:
        // CUDA
        // TODO: Rename as 'device_vbo_buffer'
        struct cudaGraphicsResource *cuda_vbo_resource;
        void *d_vbo_buffer = NULL;
};


#endif
