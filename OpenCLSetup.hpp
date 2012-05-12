// OpenCLSetup.cpp
//
//    This is a simple wrapper that demonstrates basic OpenCL setup and
//    can be used to solve basic problems.

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <string>
#include <map>


#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <ctime>

using namespace std;

timespec diff(timespec start, timespec end) {
    timespec temp;
    if ((end.tv_nsec - start.tv_nsec) < 0) {
        temp.tv_sec = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    }
    else {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return temp;
}

// a simple wrapper that holds onto a bunch of stuff
class OpenCLWrapper {
public:
    cl_context context;
    cl_command_queue commandQueue;
    cl_program program;
    cl_device_id device;
    map<string, cl_kernel> kernels;
    vector<cl_mem> memObjects;
    cl_int errNum;

    bool enableProfiling;

    map<int, cl_event> events;

    OpenCLWrapper() {
        context = 0;
        commandQueue = 0;
        program = 0;
        device = 0;
        errNum = 0;
        enableProfiling = false;
    }

    ///
    //  Create an OpenCL context on the first available platform using
    //  either a GPU or CPU depending on what is available.
    //
    void createContext() {
        cl_uint numPlatforms;
        cl_platform_id firstPlatformId;

        // First, select an OpenCL platform to run on.  For this example, we
        // simply choose the first available platform.  Normally, you would
        // query for all available platforms and select the most appropriate one.
        errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
        if (errNum != CL_SUCCESS || numPlatforms <= 0) {
            throw runtime_error("Failed to find any OpenCL platforms.");
        }

        // Next, create an OpenCL context on the platform.  Attempt to
        // create a GPU-based context, and if that fails, try to create
        // a CPU-based context.
        cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM,
                (cl_context_properties) firstPlatformId, 0 };
        context = clCreateContextFromType(contextProperties,
                CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS) {
            std::cout << "Could not create GPU context, trying CPU..."
                    << std::endl;
            context = clCreateContextFromType(contextProperties,
                    CL_DEVICE_TYPE_CPU, NULL, NULL, &errNum);
            if (errNum != CL_SUCCESS) {
                throw runtime_error("Failed to create an OpenCL GPU or CPU context.");
            }
        }
    }

    ///
    //  Create a command queue on the first device available on the
    //  context
    //
    void createCommandQueue() {
        cl_device_id *devices;
        size_t deviceBufferSize = -1;

        // First get the size of the devices buffer
        errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL,
                &deviceBufferSize);
        if (errNum != CL_SUCCESS) {
            throw runtime_error("Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)");
        }

        if (deviceBufferSize <= 0) {
            throw runtime_error("No devices available.");
        }

        // Allocate memory for the devices buffer
        devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
        errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES,
                deviceBufferSize, devices, NULL);
        if (errNum != CL_SUCCESS) {
            delete[] devices;
            throw runtime_error("Failed to get device IDs");
        }

        // In this example, we just choose the first available device.  In a
        // real program, you would likely use all available devices or choose
        // the highest performance device based on OpenCL device queries
        commandQueue = clCreateCommandQueue(context, devices[0],
                enableProfiling ? CL_QUEUE_PROFILING_ENABLE : 0, NULL);
        if (commandQueue == NULL) {
            delete[] devices;
            throw runtime_error("Failed to create commandQueue for device 0");
        }

        device = devices[0];
        delete[] devices;
    }

    ///
    //  Create an OpenCL program from the kernel source file
    //
    void createProgram(const char* fileName) {
        std::ifstream kernelFile(fileName, std::ios::in);
        if (!kernelFile.is_open()) {
            throw runtime_error("Failed to open file for reading: ");
        }

        std::ostringstream oss;
        oss << kernelFile.rdbuf();

        std::string srcStdStr = oss.str();
        const char *srcStr = srcStdStr.c_str();
        program = clCreateProgramWithSource(context, 1, (const char**) &srcStr,
                NULL, NULL);
        if (program == NULL) {
            throw runtime_error("Failed to create CL program from source.");
        }

        errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        if (errNum != CL_SUCCESS) {
            // Determine the reason for the error
            char buildLog[16384];
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                    sizeof(buildLog), buildLog, NULL);

            clReleaseProgram(program);
            throw runtime_error(string("Error in kernel: ") + string(buildLog));
        }
    }

    void createKernel(const char* functionName) {
        cl_kernel kernel = clCreateKernel(program, functionName, NULL);
        if (kernel == NULL) {
           throw runtime_error("Failed to create kernel");
        }
        else {
            kernels[string(functionName)] = kernel;
        }
    }

    bool addMemObject(cl_mem m) {
        memObjects.push_back(m);

        if (memObjects.back() == NULL) {
            throw runtime_error("Error creating memory objects.");
        }

        return true;
    }

    void check(int error, const char* message) {
        errNum = error;
        if (errNum != CL_SUCCESS) {
            throw runtime_error(message);
        }
    }

    ///
    //  Cleanup any created OpenCL resources
    //
    void cleanup() {

        for (int i = 0; i < memObjects.size(); i++) {
            if (memObjects[i] != 0)
                clReleaseMemObject(memObjects[i]);
        }
        if (commandQueue != 0)
            clReleaseCommandQueue(commandQueue);

        for (map<string,cl_kernel>::iterator iter = kernels.begin();
                iter != kernels.end(); iter++) {
            clReleaseKernel((*iter).second);
        }


        if (program != 0)
            clReleaseProgram(program);

        if (context != 0)
            clReleaseContext(context);

    }
};


