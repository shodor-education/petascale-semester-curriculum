
///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void simple_vbo_kernel(float4 *pos, unsigned int width, unsigned int height, float time)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // calculate uv coordinates
    float u = x / (float) width;
    float v = y / (float) height;
    u = u*2.0f - 1.0f;
    v = v*2.0f - 1.0f;

    // calculate simple sine wave pattern
    float freq = 4.0f;
    float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

    // write output vertex
    pos[y*width+x] = make_float4(u, w, v, 1.0f);
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////


void launch_kernel(float4 *pos, unsigned int mesh_width,
                   unsigned int mesh_height, float time)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    simple_vbo_kernel<<< grid, block>>>(pos, mesh_width, mesh_height, time);
}

class SDLWindow{
    public:
        //Starts up SDL, creates window, and initializes OpenGL
        bool init();

        //Initializes rendering program and clear color
        bool initGL();

        //Per frame update
        void update();

        //Renders quad to the screen
        void render();

        //Frees media and shuts down SDL
        void close();

        // Initialization function
        bool init() {
            //Initialization flag
            bool success = true;

            //Initialize SDL
            if (SDL_Init(SDL_INIT_VIDEO) < 0) {
                printf("SDL could not initialize! SDL Error: %s\n", SDL_GetError());
                success = false;
            }
            else {
                //Use OpenGL 3.3 core
                SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
                SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
                SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);


                //Create window
                gWindow = SDL_CreateWindow("Lab Setup", 
                                            SDL_WINDOWPOS_UNDEFINED, 
                                            SDL_WINDOWPOS_UNDEFINED, 
                                            SCREEN_WIDTH, 
                                            SCREEN_HEIGHT, 
                                            SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
                // If we fail to create a window, then record some error
                // information.
                if (gWindow == NULL) {
                    printf("Window could not be created! SDL Error: %s\n", SDL_GetError());
                    success = false;
                }
                else {
                    //Create context
                    gContext = SDL_GL_CreateContext(gWindow);
                    if (gContext == NULL) {
                        printf("OpenGL context could not be created! SDL Error: %s\n", SDL_GetError());
                        success = false;
                    }
                    else {

                        // Initialize GLAD Library
                        gladLoadGLLoader(SDL_GL_GetProcAddress);
                        printf("Vendor: %s\n", glGetString(GL_VENDOR));
                        printf("Renderer: %s\n", glGetString(GL_RENDERER));
                        printf("Version: %s\n", glGetString(GL_VERSION));
                        printf("Shading language: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

                        //Use Vsync
                        if (SDL_GL_SetSwapInterval(1) < 0) {
                            printf("Warning: Unable to set VSync! SDL Error: %s\n", SDL_GetError());
                        }

                        //Initialize OpenGL
                        if (!initGL()) {
                            printf("Unable to initialize OpenGL!\n");
                            success = false;
                        }
                    }
                }
            }

            return success;
        }

        // This function initializes OpenGL
        // Typically we are setting up OpenGL, and any shader programs here.
        bool initGL() {
            //Success flag
            bool success = true;

            //Generate program
            gProgramID = glCreateProgram();

            //Create vertex shader
            GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);

            //Get vertex source
            const GLchar* vertexShaderSource[] = {
                "#version 140\nin vec2 LVertexPos2D; void main() { gl_Position = vec4( LVertexPos2D.x, LVertexPos2D.y, 0, 1 ); }"
            };

            //Set vertex source
            glShaderSource(vertexShader, 1, vertexShaderSource, NULL);

            //Compile vertex source
            glCompileShader(vertexShader);

            //Check vertex shader for errors
            GLint vShaderCompiled = GL_FALSE;
            glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &vShaderCompiled);
            if (vShaderCompiled != GL_TRUE) {
                printf("Unable to compile vertex shader %d!\n", vertexShader);
                printShaderLog(vertexShader);
                success = false;
            }
            else {
                //Attach vertex shader to program
                glAttachShader(gProgramID, vertexShader);

                //Create fragment shader
                GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

                //Get fragment source
                const GLchar* fragmentShaderSource[] = {
                    "#version 140\nout vec4 LFragment; void main() { LFragment = vec4( 1.0, 1.0, 1.0, 1.0 ); }"
                };

                //Set fragment source
                glShaderSource(fragmentShader, 1, fragmentShaderSource, NULL);

                //Compile fragment source
                glCompileShader(fragmentShader);

                //Check fragment shader for errors
                GLint fShaderCompiled = GL_FALSE;
                glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &fShaderCompiled);
                if (fShaderCompiled != GL_TRUE) {
                    printf("Unable to compile fragment shader %d!\n", fragmentShader);
                    printShaderLog(fragmentShader);
                    success = false;
                }
                else {
                    //Attach fragment shader to program
                    glAttachShader(gProgramID, fragmentShader);

                    //Link program
                    glLinkProgram(gProgramID);

                    //Check for errors
                    GLint programSuccess = GL_TRUE;
                    glGetProgramiv(gProgramID, GL_LINK_STATUS, &programSuccess);
                    if (programSuccess != GL_TRUE) {
                        printf("Error linking program %d!\n", gProgramID);
                        printProgramLog(gProgramID);
                        success = false;
                    }
                    else {
                        //Get vertex attribute location
                        gVertexPos2DLocation = glGetAttribLocation(gProgramID, "LVertexPos2D");
                        if (gVertexPos2DLocation == -1) {
                            printf("LVertexPos2D is not a valid glsl program variable!\n");
                            success = false;
                        }
                        else {
                            //Initialize clear color
                            glClearColor(1.f, 0.f, 0.f, 1.f);

                            //VBO data
                            GLfloat vertexData[] = {
                                -0.5f, -0.5f,
                                0.5f, -0.5f,
                                0.5f,  0.5f,
                                -0.5f,  0.5f
                            };

                            //IBO data
                            GLuint indexData[] = { 0, 1, 2, 3 };

                            //Create VBO
                            glGenBuffers(1, &gVBO);
                            glBindBuffer(GL_ARRAY_BUFFER, gVBO);
                            glBufferData(GL_ARRAY_BUFFER, 2 * 4 * sizeof(GLfloat), vertexData, GL_STATIC_DRAW);

                            //Create IBO
                            glGenBuffers(1, &gIBO);
                            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gIBO);
                            glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 * sizeof(GLuint), indexData, GL_STATIC_DRAW);
                        }
                    }
                }
            }

            return success;
        }

        // This is where we do work in our graphics applications
        // that is constantly refreshed.
        void update() {
            //No per frame update needed
        }

        // This function draws images.
        void render() {
            // Clear color buffer
            // This makes the screen go "black" or whatever color we choose.
            glClear(GL_COLOR_BUFFER_BIT);

            // Render quad
            if (gRenderQuad) {
                //Bind program
                glUseProgram(gProgramID);

                //Enable vertex position
                glEnableVertexAttribArray(gVertexPos2DLocation);

                //Set vertex data
                glBindBuffer(GL_ARRAY_BUFFER, gVBO);
                glVertexAttribPointer(gVertexPos2DLocation, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), NULL);

                //Set index data and render
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gIBO);
                glDrawElements(GL_TRIANGLE_FAN, 4, GL_UNSIGNED_INT, NULL);

                //Disable vertex position
                glDisableVertexAttribArray(gVertexPos2DLocation);

                //Unbind program
                glUseProgram(NULL);
            }
        }

        // This function is called when we quit SDL
        void close() {
            //Deallocate program
            glDeleteProgram(gProgramID);

            //Destroy window	
            SDL_DestroyWindow(gWindow);
            gWindow = NULL;

            //Quit SDL subsystems
            SDL_Quit();
        }

        // Prints information about our program.
        // This is useful for debugging.
        void printProgramLog(GLuint program) {
            //Make sure name is shader
            if (glIsProgram(program)) {
                //Program log length
                int infoLogLength = 0;
                int maxLength = infoLogLength;

                //Get info string length
                glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);

                //Allocate string
                char* infoLog = new char[maxLength];

                //Get info log
                glGetProgramInfoLog(program, maxLength, &infoLogLength, infoLog);
                if (infoLogLength > 0) {
                    //Print Log
                    printf("%s\n", infoLog);
                }

                //Deallocate string
                delete[] infoLog;
            }
            else {
                printf("Name %d is not a program\n", program);
            }
        }
        //Shader loading utility programs
        void printProgramLog(GLuint program);
        void printShaderLog(GLuint shader);








        // Prints information about our shaders.
        // This is useful for debugging.
        void printShaderLog(GLuint shader) {
            //Make sure name is shader
            if (glIsShader(shader)) {
                //Shader log length
                int infoLogLength = 0;
                int maxLength = infoLogLength;

                //Get info string length
                glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);

                //Allocate string
                char* infoLog = new char[maxLength];

                //Get info log
                glGetShaderInfoLog(shader, maxLength, &infoLogLength, infoLog);
                if (infoLogLength > 0) {
                    //Print Log
                    printf("%s\n", infoLog);
                }

                //Deallocate string
                delete[] infoLog;
            }
            else {
                printf("Name %d is not a shader\n", shader);
            }
        }
    private:

        //The window we'll be rendering to
        SDL_Window* gWindow = NULL;

        //OpenGL context
        SDL_GLContext gContext;
        //Render flag
        bool gRenderQuad = true;

        // Store information we want to ship to
        // the GPU.
        GLuint gProgramID = 0;
        GLint gVertexPos2DLocation = -1;
        GLuint gVBO = 0;
        GLuint gIBO = 0;
        //Screen dimension constants
        const int SCREEN_WIDTH = 1080;
        const int SCREEN_HEIGHT = 720;

};

class CudaEnabledVertexBufferObject{
    public:
        void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
                       unsigned int vbo_res_flags)
        {
            assert(vbo);

            // create buffer object
            glGenBuffers(1, vbo);
            glBindBuffer(GL_ARRAY_BUFFER, *vbo);

            // initialize buffer object
            unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
            glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

            glBindBuffer(GL_ARRAY_BUFFER, 0);

            // register this buffer object with CUDA
            checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

            SDK_CHECK_ERROR_GL();
        }
        ////////////////////////////////////////////////////////////////////////////////
        //! Delete VBO
        ////////////////////////////////////////////////////////////////////////////////
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
}