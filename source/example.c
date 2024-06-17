#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdalign.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <float.h>
#include <limits.h>

#include <Windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>

enum
{
    // Maximum number of hardware adapters
    MAX_HARDWARE_ADAPTER_COUNT = 16,

    // Test data element count
    TEST_DATA_COUNT = 4096,

    // Pre-compute execution signal value
    UPLOAD_DATA_SIGNAL_VALUE = 1,

    // Compute execution complete signal value
    COMPUTE_EXECUTED_SIGNAL_VALUE = 2
};

// The factory to create the device
static IDXGIFactory4* s_factory = NULL;

// The compatible D3D12 device object
static ID3D12Device* s_device;

// The root signature for compute pipeline state object
static ID3D12RootSignature* s_computeRootSignature;

// The compute pipeline state object
static ID3D12PipelineState* s_computeState;

// The descriptor heap resource object. 
// In this sample, there're two slots in this heap. 
// The first slot stores the shader view resource descriptor, 
// and the second slot stores the unordered access view descriptor.
static ID3D12DescriptorHeap* s_heap;

// The destination buffer object with unordered access view type
static ID3D12Resource* s_dstDataBuffer;

// The source buffer object with shader source view type
static ID3D12Resource* s_srcDataBuffer;

// The intermediate buffer object used to copy the source data to the SRV buffer
static ID3D12Resource* s_uploadBuffer;

// The heap descriptor(of SRV, UAV and CBV type)  size
static size_t s_srvUavDescriptorSize;

// The command allocator object
static ID3D12CommandAllocator* s_computeAllocator;

// The command queue object
static ID3D12CommandQueue* s_computeCommandQueue;

// The command list object
static ID3D12GraphicsCommandList* s_computeCommandList;

// The fence object
static ID3D12Fence* s_fence;

// Windows API event handle
static HANDLE s_eventHandle;

static int s_DataBuffer0[TEST_DATA_COUNT];

static void TransWStrToString(char dstBuf[], const WCHAR srcBuf[])
{
    if (dstBuf == NULL || srcBuf == NULL) return;

    const int len = WideCharToMultiByte(CP_UTF8, 0, srcBuf, -1, NULL, 0, NULL, NULL);
    WideCharToMultiByte(CP_UTF8, 0, srcBuf, -1, dstBuf, len, NULL, NULL);
    dstBuf[len] = '\0';
}

// Updates subresources, all the subresource arrays should be populated.
// This function is the C-style implementation translated from C++ style inline function in the D3DX12 library.
static void WriteDeviceResourceAndSync(
    _In_ ID3D12GraphicsCommandList* commandList,
    _In_ ID3D12Resource* pDestinationDeviceResource,
    _In_ ID3D12Resource* pUploadHostResource,
    size_t dstOffset,
    size_t srcoffset,
    size_t dataSize)
{
    const D3D12_RESOURCE_BARRIER beginCopyBarrier = {
        .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
        .Transition = {
            .pResource = pDestinationDeviceResource,
            .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            .StateBefore = D3D12_RESOURCE_STATE_COMMON,
            .StateAfter = D3D12_RESOURCE_STATE_COPY_DEST
        }
    };
    commandList->lpVtbl->ResourceBarrier(commandList, 1, &beginCopyBarrier);

    commandList->lpVtbl->CopyBufferRegion(commandList, pDestinationDeviceResource, (UINT64)dstOffset, pUploadHostResource, srcoffset, dataSize);

    const D3D12_RESOURCE_BARRIER endCopyBarrier = {
        .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
        .Transition = {
            .pResource = pDestinationDeviceResource,
            .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            .StateBefore = D3D12_RESOURCE_STATE_COPY_DEST,
            .StateAfter = D3D12_RESOURCE_STATE_GENERIC_READ
        }
    };
    commandList->lpVtbl->ResourceBarrier(commandList, 1, &endCopyBarrier);
}

static void SyncAndReadDeviceResource(
    _In_ ID3D12GraphicsCommandList* commandList,
    _In_ ID3D12Resource* pReadbackHostResource,
    _In_ ID3D12Resource* pSourceDeviceResource)
{
    const D3D12_RESOURCE_BARRIER beginCopyBarrier = {
        .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
        .Transition = {
            .pResource = pSourceDeviceResource,
            .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            .StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            .StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE
        }
    };
    commandList->lpVtbl->ResourceBarrier(commandList, 1, &beginCopyBarrier);

    commandList->lpVtbl->CopyResource(commandList, pReadbackHostResource, pSourceDeviceResource);

    const D3D12_RESOURCE_BARRIER endCopyBarrier = {
        .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
        .Transition = {
            .pResource = pSourceDeviceResource,
            .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            .StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE,
            .StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        }
    };
    commandList->lpVtbl->ResourceBarrier(commandList, 1, &endCopyBarrier);
}

static bool CreateD3D12Device(bool useWARPAdapter)
{
    HRESULT hRes = S_OK;

#if defined(DEBUG) || defined(_DEBUG)
    // Enable debug mode
    ID3D12Debug* debugController = NULL;
    hRes = D3D12GetDebugInterface(&IID_ID3D12Debug, (void**)&debugController);
    if (SUCCEEDED(hRes)) {
        debugController->lpVtbl->EnableDebugLayer(debugController);
    }
    else {
        printf("WARNING: D3D12GetDebugInterface enable failed: %ld\n", hRes);
    }
#endif // DEBUG

    hRes = CreateDXGIFactory1(&IID_IDXGIFactory4, (void**)&s_factory);
    if (FAILED(hRes))
    {
        fprintf(stderr, "CreateDXGIFactory1 failed: %ld\n", hRes);
        return false;
    }

    if (useWARPAdapter)
    {
        // Here, we shall use a WARP (Windows Advanced Rasterization Platform) adapter
        IDXGIAdapter* warpAdapter = NULL;
        hRes = s_factory->lpVtbl->EnumWarpAdapter(s_factory, &IID_IDXGIAdapter, (void**)&warpAdapter);
        if (FAILED(hRes))
        {
            fprintf(stderr, "EnumWarpAdapter failed: %ld\n", hRes);
            return false;
        }

        // Create the D3D12 device
        hRes = D3D12CreateDevice((IUnknown*)warpAdapter, D3D_FEATURE_LEVEL_12_0, &IID_ID3D12Device, (void**)&s_device);
        if (FAILED(hRes))
        {
            fprintf(stderr, "D3D12CreateDevice failed: %ld\n", hRes);
            return false;
        }
    }
    else
    {
        // Enumerate the adapters (video cards)
        IDXGIAdapter1* hardwareAdapters[MAX_HARDWARE_ADAPTER_COUNT] = { 0 };
        UINT foundAdapterCount;
        for (foundAdapterCount = 0; foundAdapterCount < MAX_HARDWARE_ADAPTER_COUNT; ++foundAdapterCount)
        {
            hRes = s_factory->lpVtbl->EnumAdapters1(s_factory, foundAdapterCount, &hardwareAdapters[foundAdapterCount]);
            if (FAILED(hRes))
            {
                if (hRes != DXGI_ERROR_NOT_FOUND) {
                    printf("WARNING: Some error occurred during enumerating adapters: %ld\n", hRes);
                }
                break;
            }
        }
        if (foundAdapterCount == 0)
        {
            fprintf(stderr, "There are no Direct3D capable adapters found on the current platform...\n");
            return false;
        }

        printf("Found %u Direct3D capable device%s in all.\n", foundAdapterCount, foundAdapterCount > 1 ? "s" : "");

        DXGI_ADAPTER_DESC1 adapterDesc = { 0 };
        char strBuf[512] = { '\0' };
        for (UINT i = 0; i < foundAdapterCount; ++i)
        {
            hRes = hardwareAdapters[i]->lpVtbl->GetDesc1(hardwareAdapters[i], &adapterDesc);
            if (FAILED(hRes))
            {
                fprintf(stderr, "hardwareAdapters[%u] GetDesc1 failed: %ld\n", i, hRes);
                return false;
            }

            TransWStrToString(strBuf, adapterDesc.Description);
            printf("Adapter[%u]: %s\n", i, strBuf);
        }
        printf("Please Choose which adapter to use: ");

        gets_s(strBuf, sizeof(strBuf));

        char* endChar = NULL;
        int selectedAdapterIndex = atoi(strBuf);
        if (selectedAdapterIndex < 0 || selectedAdapterIndex >= (int)foundAdapterCount)
        {
            puts("WARNING: The index you input exceeds the range of available adatper count. So adatper[0] will be used!");
            selectedAdapterIndex = 0;
        }

        hRes = hardwareAdapters[selectedAdapterIndex]->lpVtbl->GetDesc1(hardwareAdapters[selectedAdapterIndex], &adapterDesc);
        if (FAILED(hRes))
        {
            fprintf(stderr, "hardwareAdapters[%d] GetDesc1 failed: %ld\n", selectedAdapterIndex, hRes);
            return false;
        }

        TransWStrToString(strBuf, adapterDesc.Description);

        printf("\nYou have chosen adapter[%ld]\n", selectedAdapterIndex);
        printf("Adapter description: %s\n", strBuf);
        printf("Dedicated Video Memory: %.1f GB\n", (double)(adapterDesc.DedicatedVideoMemory) / (1024.0 * 1024.0 * 1024.0));
        printf("Dedicated System Memory: %.1f GB\n", (double)(adapterDesc.DedicatedSystemMemory) / (1024.0 * 1024.0 * 1024.0));
        printf("Shared System Memory: %.1f GB\n", (double)(adapterDesc.SharedSystemMemory) / (1024.0 * 1024.0 * 1024.0));

        hRes = D3D12CreateDevice((IUnknown*)hardwareAdapters[selectedAdapterIndex], D3D_FEATURE_LEVEL_12_0, &IID_ID3D12Device, (void**)&s_device);
        if (FAILED(hRes))
        {
            fprintf(stderr, "D3D12CreateDevice failed: %ld\n", hRes);
            return false;
        }
    }

    D3D12_FEATURE_DATA_SHADER_MODEL shaderModel = { .HighestShaderModel = D3D_HIGHEST_SHADER_MODEL };
    hRes = s_device->lpVtbl->CheckFeatureSupport(s_device, D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel));
    if (FAILED(hRes))
    {
        fprintf(stderr, "CheckFeatureSupport for `D3D12_FEATURE_SHADER_MODEL` failed: %ld\n", hRes);
        return false;
    }

    const int minor = shaderModel.HighestShaderModel & 0x0f;
    const int major = shaderModel.HighestShaderModel >> 4;
    printf("Current device support highest shader model: %d.%d\n", major, minor);

    D3D12_FEATURE_DATA_ROOT_SIGNATURE rootSignature = { .HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1 };
    hRes = s_device->lpVtbl->CheckFeatureSupport(s_device, D3D12_FEATURE_ROOT_SIGNATURE, &rootSignature, sizeof(rootSignature));
    if (FAILED(hRes))
    {
        fprintf(stderr, "CheckFeatureSupport for `D3D12_FEATURE_DATA_ROOT_SIGNATURE` failed: %ld\n", hRes);
        return false;
    }

    const char* signatureVersion = "1.0";
    switch (rootSignature.HighestVersion)
    {
    case D3D_ROOT_SIGNATURE_VERSION_1_0:
    default:
        break;

    case D3D_ROOT_SIGNATURE_VERSION_1_1:
        signatureVersion = "1.1";
        break;
    }
    printf("Current device supports highest root signature version: %s\n", signatureVersion);

    // Check 4X MSAA quality support for our back buffer format.
    // All Direct3D 11 capable devices support 4X MSAA for all render 
    // target formats, so we only need to check quality support.
    // This step is optional.
    D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS msQualityLevels = {
        .Format = DXGI_FORMAT_R8G8B8A8_UNORM,
        .SampleCount = 4,
        .Flags = D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE,
        .NumQualityLevels = 0
    };
    hRes = s_device->lpVtbl->CheckFeatureSupport(s_device, D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &msQualityLevels, sizeof(msQualityLevels));
    if (FAILED(hRes))
    {
        fprintf(stderr, "D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS check failed: %ld\n", hRes);
        return false;
    }
    printf("msaaQuality: %u\n", msQualityLevels.NumQualityLevels);

    puts("\n================================================\n");

    return true;
}

static bool CreateRootSignature(void)
{
    const D3D12_DESCRIPTOR_RANGE ranges[] = {
        {
            .RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            .NumDescriptors = 1,
            .BaseShaderRegister = 0,
            .RegisterSpace = 0,
            .OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND
        },
        {
            .RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
            .NumDescriptors = 1,
            .BaseShaderRegister = 0,
            .RegisterSpace = 0,
            .OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND
        }
    };

    const D3D12_ROOT_PARAMETER rootParameters[] = {
        {
            .ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            .DescriptorTable = {.NumDescriptorRanges = 1, .pDescriptorRanges = &ranges[0] },
            .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL
        },
        {
            .ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            .DescriptorTable = {.NumDescriptorRanges = 1, .pDescriptorRanges = &ranges[1] },
            .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL
        }
    };

    const D3D12_ROOT_SIGNATURE_DESC computeRootSignatureDesc = {
        .NumParameters = (UINT)(sizeof(rootParameters) / sizeof(rootParameters[0])),
        .pParameters = rootParameters,
        .NumStaticSamplers = 0,
        .Flags = D3D12_ROOT_SIGNATURE_FLAG_DENY_VERTEX_SHADER_ROOT_ACCESS |
                    D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
                    D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
                    D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS |
                    D3D12_ROOT_SIGNATURE_FLAG_DENY_AMPLIFICATION_SHADER_ROOT_ACCESS | D3D12_ROOT_SIGNATURE_FLAG_DENY_MESH_SHADER_ROOT_ACCESS
    };

    ID3DBlob* signature = NULL;
    ID3DBlob* error = NULL;
    HRESULT hRes = D3D12SerializeRootSignature(&computeRootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error);
    do
    {
        if (FAILED(hRes))
        {
            fprintf(stderr, "Failed to serialize versioned root signature: %ld\n", hRes);
            break;
        }

        hRes = s_device->lpVtbl->CreateRootSignature(s_device, 0, signature->lpVtbl->GetBufferPointer(signature),
            signature->lpVtbl->GetBufferSize(signature), &IID_ID3D12RootSignature, &s_computeRootSignature);
        if (FAILED(hRes))
        {
            fprintf(stderr, "Failed to create root signature: %ld\n", hRes);
            break;
        }
    } while (false);

    if (signature != NULL) {
        signature->lpVtbl->Release(signature);
    }
    if (error != NULL) {
        error->lpVtbl->Release(error);
    }

    if (FAILED(hRes)) return false;

    // This setting is optional.
    hRes = s_computeRootSignature->lpVtbl->SetName(s_computeRootSignature, L"s_computeRootSignature");
    if (FAILED(hRes))
    {
        fprintf(stderr, "s_computeRootSignature setName failed: %ld\n", hRes);
        return false;
    }

    return true;
}

// Create the Shader Resource View buffer object
static ID3D12Resource* CreateSRVBuffer(const void* inputData, size_t dataSize, UINT nElems, UINT elemSize)
{
    ID3D12Resource* resultBuffer = NULL;
    HRESULT hr = S_OK;

    do
    {
        const D3D12_HEAP_PROPERTIES heapProperties = {
            .Type = D3D12_HEAP_TYPE_DEFAULT,
            .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            .MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,
            .CreationNodeMask = 1,
            .VisibleNodeMask = 1
        };
        const D3D12_HEAP_PROPERTIES heapUploadProperties = {
            .Type = D3D12_HEAP_TYPE_UPLOAD,
            .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            .MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,
            .CreationNodeMask = 1,
            .VisibleNodeMask = 1
        };

        const D3D12_RESOURCE_DESC resourceDesc = {
            .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
            .Alignment = 0,
            .Width = dataSize,
            .Height = 1,
            .DepthOrArraySize = 1,
            .MipLevels = 1,
            .Format = DXGI_FORMAT_UNKNOWN,
            .SampleDesc = {.Count = 1, .Quality = 0 },
            .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            .Flags = D3D12_RESOURCE_FLAG_NONE
        };
        const D3D12_RESOURCE_DESC uploadBufferDesc = {
            .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
            .Alignment = 0,
            .Width = dataSize,
            .Height = 1,
            .DepthOrArraySize = 1,
            .MipLevels = 1,
            .Format = DXGI_FORMAT_UNKNOWN,
            .SampleDesc = {.Count = 1, .Quality = 0 },
            .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            .Flags = D3D12_RESOURCE_FLAG_NONE
        };

        // Create the SRV buffer and make it as the copy destination.
        hr = s_device->lpVtbl->CreateCommittedResource(s_device, &heapProperties, D3D12_HEAP_FLAG_NONE, &resourceDesc,
            D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource, (void**)&resultBuffer);
        if (FAILED(hr))
        {
            fprintf(stderr, "CreateCommittedResource for resultBuffer failed: %ld\n", hr);
            break;
        }

        // Create the upload buffer and make it as the generic read intermediate.
        hr = s_device->lpVtbl->CreateCommittedResource(s_device, &heapUploadProperties, D3D12_HEAP_FLAG_NONE, &uploadBufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ, NULL, &IID_ID3D12Resource, (void**)&s_uploadBuffer);
        if (FAILED(hr))
        {
            fprintf(stderr, "CreateCommittedResource for s_uploadBuffer failed: %ld\n", hr);
            break;
        }

        // Transfer data from host to the device UAV buffer
        void* hostMemPtr = NULL;
        const D3D12_RANGE readRange = { 0, 0 };
        hr = s_uploadBuffer->lpVtbl->Map(s_uploadBuffer, 0, &readRange, &hostMemPtr);
        if (FAILED(hr))
        {
            fprintf(stderr, "Map vertex buffer failed: %ld\n", hr);
            break;
        }

        memcpy(hostMemPtr, inputData, dataSize);
        s_uploadBuffer->lpVtbl->Unmap(s_uploadBuffer, 0, NULL);

        WriteDeviceResourceAndSync(s_computeCommandList, resultBuffer, s_uploadBuffer, 0, 0, dataSize);

        // Attention! None of the operations above has been executed.
        // They have just been put into the command list.
        // So the intermediate buffer s_uploadBuffer MUST NOT be released here.

        // Setup the SRV descriptor. This will be stored in the first slot of the heap.
        const D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {
            .Format = DXGI_FORMAT_UNKNOWN,
            .ViewDimension = D3D12_SRV_DIMENSION_BUFFER,
            .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            .Buffer = {
                .FirstElement = 0,
                .NumElements = nElems,
                .StructureByteStride = elemSize,
                .Flags = D3D12_BUFFER_SRV_FLAG_NONE
            }
        };

        // Get the descriptor handle from the descriptor heap.
        D3D12_CPU_DESCRIPTOR_HANDLE srvHandle = { 0 };
        s_heap->lpVtbl->GetCPUDescriptorHandleForHeapStart(s_heap, &srvHandle);
        // srvHandle will occupy the first slot, so `srvHandle.ptr += 0 * s_srvUavDescriptorSize;`

        // Create the SRV for the buffer with the descriptor handle
        s_device->lpVtbl->CreateShaderResourceView(s_device, resultBuffer, &srvDesc, srvHandle);
    } while (false);

    if (FAILED(hr))
    {
        fprintf(stderr, "CreateSRVBuffer failed: %ld\n", hr);
        return NULL;
    }

    return resultBuffer;
}

// Create the Unordered Access View buffer object
static ID3D12Resource* CreateUAV_RWBuffer(const void* inputData, size_t dataSize, UINT elemCount, UINT elemSize)
{
    ID3D12Resource* resultBuffer = NULL;
    HRESULT hr = S_OK;

    do
    {
        const D3D12_HEAP_PROPERTIES heapProperties = {
            .Type = D3D12_HEAP_TYPE_DEFAULT,
            .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            .MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,
            .CreationNodeMask = 1,
            .VisibleNodeMask = 1
        };
        const D3D12_RESOURCE_DESC resourceDesc = {
            .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
            .Alignment = 0,
            .Width = dataSize,
            .Height = 1,
            .DepthOrArraySize = 1,
            .MipLevels = 1,
            .Format = DXGI_FORMAT_UNKNOWN,
            .SampleDesc = {.Count = 1, .Quality = 0 },
            .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        };

        // Create the UAV buffer and make it in the unordered access state.
        hr = s_device->lpVtbl->CreateCommittedResource(s_device, &heapProperties, D3D12_HEAP_FLAG_NONE, &resourceDesc,
            D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource, (void**)&resultBuffer);

        if (FAILED(hr))
        {
            fprintf(stderr, "Failed to create resultBuffer: %ld\n", hr);
            return NULL;
        }

        // Setup the UAV descriptor. This will be stored in the second slot of the heap.
        const D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {
            .Format = DXGI_FORMAT_UNKNOWN,
            .ViewDimension = D3D12_UAV_DIMENSION_BUFFER,
            .Buffer = {
                .FirstElement = 0,
                .NumElements = elemCount,
                .StructureByteStride = elemSize,
                .CounterOffsetInBytes = 0,
                .Flags = D3D12_BUFFER_UAV_FLAG_NONE
            }
        };

        // Get the descriptor handle from the descriptor heap.
        D3D12_CPU_DESCRIPTOR_HANDLE uavHandle = { 0 };
        s_heap->lpVtbl->GetCPUDescriptorHandleForHeapStart(s_heap, &uavHandle);
        // uavHandle will occupy the second slot.
        uavHandle.ptr += 1 * s_srvUavDescriptorSize;

        s_device->lpVtbl->CreateUnorderedAccessView(s_device, resultBuffer, NULL, &uavDesc, uavHandle);

    } while (false);

    if (FAILED(hr))
    {
        fprintf(stderr, "Create UAV Buffer failed: %ld\n", hr);
        return NULL;
    }

    return resultBuffer;
}

// Create the descriptor heap and compute pipeline state object
static bool CreateComputePipelineStateObject(void)
{
    // ---- Create descriptor heaps. ----
    const D3D12_DESCRIPTOR_HEAP_DESC srvUavHeapDesc = {
        // There are two descriptors for the heap. One for SRV buffer, the other for UAV buffer
        .NumDescriptors = 2,
        .Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
        .NodeMask = 0
    };
    HRESULT hr = s_device->lpVtbl->CreateDescriptorHeap(s_device, &srvUavHeapDesc, &IID_ID3D12DescriptorHeap, (void**)&s_heap);
    if (FAILED(hr))
    {
        fprintf(stderr, "Failed to create s_srvHeap: %ld\n", hr);
        return false;
    }

    // Optionally set the name
    s_heap->lpVtbl->SetName(s_heap, L"s_heap");
    // Get the size of each descriptor handle
    s_srvUavDescriptorSize = s_device->lpVtbl->GetDescriptorHandleIncrementSize(s_device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // Create the pipeline states, which includes compiling and loading shaders.

    ID3DBlob* computeShader = NULL;
    ID3DBlob* error = NULL;

    // Enable better shader debugging with the graphics debugging tools.
    uint32_t compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;

    // Load and compile the compute shader.
    // The comppute shader file 'compute.hlsl' is just located in the current working directory.
    hr = D3DCompileFromFile(L"compute.hlsl", NULL, NULL, "CSMain", "cs_5_0", compileFlags, 0, &computeShader, &error);
    do
    {
        if (FAILED(hr))
        {
            fprintf(stderr, "D3DCompileFromFile failed: %ld\n", hr);
            if (error != NULL) {
                fprintf(stderr, "Error message: %s\n", (char*)error->lpVtbl->GetBufferPointer(error));
            }
            break;
        }

        // Describe and create the compute pipeline state object (PSO).
        const D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc = {
            .pRootSignature = s_computeRootSignature,
            .CS = (D3D12_SHADER_BYTECODE){
                .pShaderBytecode = computeShader->lpVtbl->GetBufferPointer(computeShader),
                .BytecodeLength = computeShader->lpVtbl->GetBufferSize(computeShader)
            },
            .NodeMask = 0,
            .CachedPSO = {.pCachedBlob = NULL, .CachedBlobSizeInBytes = 0 },
            .Flags = D3D12_PIPELINE_STATE_FLAG_NONE
        };
        hr = s_device->lpVtbl->CreateComputePipelineState(s_device, &computePsoDesc, &IID_ID3D12PipelineState, (void**)&s_computeState);
        if (FAILED(hr))
        {
            fprintf(stderr, "CreateComputePipelineState failed: %ld\n", hr);
            break;
        }
    } while (false);

    if (computeShader != NULL) {
        computeShader->lpVtbl->Release(computeShader);
    }
    if (error != NULL) {
        error->lpVtbl->Release(error);
    }

    if (FAILED(hr)) return false;

    return true;
}

// Create the command list and the command queue
static bool CreateComputeCommands(void)
{
    const D3D12_COMMAND_QUEUE_DESC queueDesc = {
        .Type = D3D12_COMMAND_LIST_TYPE_DIRECT,
        .Priority = 0,
        .Flags = D3D12_COMMAND_QUEUE_FLAG_NONE,
        .NodeMask = 0
    };
    HRESULT hRes = s_device->lpVtbl->CreateCommandQueue(s_device, &queueDesc, &IID_ID3D12CommandQueue, (void**)&s_computeCommandQueue);
    if (FAILED(hRes))
    {
        fprintf(stderr, "CreateCommandQueue failed: %ld\n", hRes);
        return false;
    }

    hRes = s_device->lpVtbl->CreateCommandAllocator(s_device, D3D12_COMMAND_LIST_TYPE_DIRECT, &IID_ID3D12CommandAllocator, (void**)&s_computeAllocator);
    if (FAILED(hRes))
    {
        fprintf(stderr, "CreateCommandAllocator failed: %ld\n", hRes);
        return false;
    }

    hRes = s_device->lpVtbl->CreateCommandList(s_device, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, s_computeAllocator, NULL, &IID_ID3D12CommandList, (void**)&s_computeCommandList);
    if (FAILED(hRes))
    {
        fprintf(stderr, "CreateCommandList failed: %ld\n", hRes);
        return false;
    }

    return true;
}

// Create the source buffer object and the destination buffer object.
// Initialize the SRV buffer object with the input buffer
static bool CreateBuffers(void)
{
    // 对数据资源做初始化
    for (int i = 0; i < TEST_DATA_COUNT; ++i) {
        s_DataBuffer0[i] = i + 1;
    }

    // Create the compute shader's constant buffer.
    const uint32_t bufferSize = (uint32_t)sizeof(s_DataBuffer0);
    s_srcDataBuffer = CreateSRVBuffer(s_DataBuffer0, bufferSize, TEST_DATA_COUNT, (UINT)sizeof(int));
    s_dstDataBuffer = CreateUAV_RWBuffer(NULL, bufferSize, TEST_DATA_COUNT, (UINT)sizeof(int));

    return true;
}

static bool CreateFenceAndEvent(void)
{
    HRESULT hRes = s_device->lpVtbl->CreateFence(s_device, 0, D3D12_FENCE_FLAG_NONE, &IID_ID3D12Fence, (void**)&s_fence);
    if (FAILED(hRes))
    {
        fprintf(stderr, "CreateFence failed: %ld\n", hRes);
        return  false;
    }

    s_eventHandle = CreateEventA(NULL, FALSE, FALSE, NULL);
    if (s_eventHandle == NULL)
    {
        const DWORD err = GetLastError();
        fprintf(stderr, "Failed to create event handle: %u\n", err);
        return false;
    }

    return true;
}

// Wait for the whole command queue completed
static void SyncCommandQueue(ID3D12CommandQueue* commandQueue, ID3D12Device* device, UINT64 signalValue)
{
    // Add an instruction to the command queue to set a new fence point.  Because we 
    // are on the GPU timeline, the new fence point won't be set until the GPU finishes
    // processing all the commands prior to this Signal().
    HRESULT hRes = commandQueue->lpVtbl->Signal(commandQueue, s_fence, signalValue);
    if (FAILED(hRes)) {
        fprintf(stderr, "Signal failed: %ld\n", hRes);
    }

    // Wait until the GPU has completed commands up to this fence point.
    // Fire event when GPU hits current fence.  
    hRes = s_fence->lpVtbl->SetEventOnCompletion(s_fence, signalValue, s_eventHandle);
    if (FAILED(hRes)) {
        fprintf(stderr, "Set event failed: %ld\n", hRes);
    }

    // Wait until the GPU hits current fence event is fired.
    WaitForSingleObject(s_eventHandle, INFINITE);
}

// Do the compute operation and fetch the result
static void DoCompute(void)
{
    ID3D12Resource* readBackBuffer = NULL;
    const D3D12_HEAP_PROPERTIES heapProperties = {
        .Type = D3D12_HEAP_TYPE_READBACK,
        .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        .MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,
        .CreationNodeMask = 1,
        .VisibleNodeMask = 1
    };
    const D3D12_RESOURCE_DESC resourceDesc = {
        .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
        .Alignment = 0,
        .Width = sizeof(s_DataBuffer0),
        .Height = 1,
        .DepthOrArraySize = 1,
        .MipLevels = 1,
        .Format = DXGI_FORMAT_UNKNOWN,
        .SampleDesc = {.Count = 1, .Quality = 0 },
        .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        .Flags = D3D12_RESOURCE_FLAG_NONE
    };

    // Create the read-back buffer object that will fetch the result from the UAV buffer object.
    // And make it as the copy destination.
    HRESULT hr = s_device->lpVtbl->CreateCommittedResource(s_device, &heapProperties, D3D12_HEAP_FLAG_NONE, &resourceDesc,
        D3D12_RESOURCE_STATE_COPY_DEST, NULL, &IID_ID3D12Resource, (void**)&readBackBuffer);

    if (FAILED(hr))
    {
        fprintf(stderr, "CreateCommittedResource for readBackBuffer failed: %ld\n", hr);
        return;
    }

    // Reuse the memory associated with command recording.
    // We can only reset when the associated command lists have finished execution on the GPU.
    hr = s_computeAllocator->lpVtbl->Reset(s_computeAllocator);
    if (FAILED(hr))
    {
        fprintf(stderr, "Reset for s_computeAllocator failed: %ld\n", hr);
        return;
    }

    // A command list can be reset after it has been added to the command queue via ExecuteCommandList.
    // Reusing the command list reuses memory.
    hr = s_computeCommandList->lpVtbl->Reset(s_computeCommandList, s_computeAllocator, s_computeState);
    if (FAILED(hr))
    {
        fprintf(stderr, "Reset for s_computeCommandList failed: %ld\n", hr);
        return;
    }

    s_computeCommandList->lpVtbl->SetComputeRootSignature(s_computeCommandList, s_computeRootSignature);

    ID3D12DescriptorHeap* const ppHeaps[] = { s_heap };
    s_computeCommandList->lpVtbl->SetDescriptorHeaps(s_computeCommandList, (UINT)(sizeof(ppHeaps) / sizeof(ppHeaps[0])), ppHeaps);

    D3D12_GPU_DESCRIPTOR_HANDLE srvHandle = { 0 };
    // Get the SRV GPU descriptor handle from the descriptor heap, occupy the first slot
    s_heap->lpVtbl->GetGPUDescriptorHandleForHeapStart(s_heap, &srvHandle);

    D3D12_GPU_DESCRIPTOR_HANDLE uavHandle = { 0 };
    // Get the UAV GPU descriptor handle from the descriptor heap
    s_heap->lpVtbl->GetGPUDescriptorHandleForHeapStart(s_heap, &uavHandle);
    // Occupy the second slot
    uavHandle.ptr += 1 * s_srvUavDescriptorSize;

    s_computeCommandList->lpVtbl->SetComputeRootDescriptorTable(s_computeCommandList, 0, srvHandle);
    s_computeCommandList->lpVtbl->SetComputeRootDescriptorTable(s_computeCommandList, 1, uavHandle);

    // Dispatch the GPU threads
    s_computeCommandList->lpVtbl->Dispatch(s_computeCommandList, 4, 1, 1);

    // Insert a barrier command to sync the dispatch operation, 
    // and make the UAV buffer object as the copy source.
    SyncAndReadDeviceResource(s_computeCommandList, readBackBuffer, s_dstDataBuffer);

    s_computeCommandList->lpVtbl->Close(s_computeCommandList);

    s_computeCommandQueue->lpVtbl->ExecuteCommandLists(s_computeCommandQueue, 1, (ID3D12CommandList * []) { (ID3D12CommandList*)s_computeCommandList });

    SyncCommandQueue(s_computeCommandQueue, s_device, COMPUTE_EXECUTED_SIGNAL_VALUE);

    void* pData;
    const D3D12_RANGE range = { 0, TEST_DATA_COUNT };
    // Map the memory buffer so that we may access the data from the host side.
    hr = readBackBuffer->lpVtbl->Map(readBackBuffer, 0, &range, &pData);
    if (FAILED(hr))
    {
        fprintf(stderr, "Map readBackBuffer failed: %ld\n", hr);
        return;
    }

    int* resultBuffer = malloc(TEST_DATA_COUNT * sizeof(*resultBuffer));
    if (resultBuffer == NULL)
    {
        fprintf(stderr, "Lack of memory for resultBuffer!\n");
        return;
    }
    memcpy(resultBuffer, pData, TEST_DATA_COUNT * sizeof(*resultBuffer));

    // After copying the data, just release the read-back buffer object.
    readBackBuffer->lpVtbl->Unmap(readBackBuffer, 0, NULL);
    readBackBuffer->lpVtbl->Release(readBackBuffer);

    // Verify the result
    bool equal = true;
    for (int i = 0; i < TEST_DATA_COUNT; i++)
    {
        if (resultBuffer[i] - 10 != s_DataBuffer0[i])
        {
            printf("%d index elements are not equal!\n", i);
            equal = false;
            break;
        }
    }
    if (equal) {
        puts("Verification OK!");
    }

    free(resultBuffer);
}

// Release all the resources
void ReleaseResources(void)
{
    if (s_eventHandle != NULL)
    {
        CloseHandle(s_eventHandle);
        s_eventHandle = NULL;
    }
    if (s_fence != NULL)
    {
        s_fence->lpVtbl->Release(s_fence);
        s_fence = NULL;
    }
    if (s_heap != NULL)
    {
        s_heap->lpVtbl->Release(s_heap);
        s_heap = NULL;
    }

    if (s_srcDataBuffer != NULL)
    {
        s_srcDataBuffer->lpVtbl->Release(s_srcDataBuffer);
        s_srcDataBuffer = NULL;
    }

    if (s_dstDataBuffer != NULL)
    {
        s_dstDataBuffer->lpVtbl->Release(s_dstDataBuffer);
        s_dstDataBuffer = NULL;
    }

    if (s_uploadBuffer != NULL)
    {
        s_uploadBuffer->lpVtbl->Release(s_uploadBuffer);
        s_uploadBuffer = NULL;
    }

    if (s_computeAllocator != NULL)
    {
        s_computeAllocator->lpVtbl->Release(s_computeAllocator);
        s_computeAllocator = NULL;
    }

    if (s_computeCommandList != NULL)
    {
        s_computeCommandList->lpVtbl->Release(s_computeCommandList);
        s_computeCommandList = NULL;
    }

    if (s_computeCommandQueue != NULL)
    {
        s_computeCommandQueue->lpVtbl->Release(s_computeCommandQueue);
        s_computeCommandQueue = NULL;
    }

    if (s_computeState != NULL)
    {
        s_computeState->lpVtbl->Release(s_computeState);
        s_computeState = NULL;
    }

    if (s_computeRootSignature != NULL)
    {
        s_computeRootSignature->lpVtbl->Release(s_computeRootSignature);
        s_computeRootSignature = NULL;
    }

    if (s_device != NULL)
    {
        s_device->lpVtbl->Release(s_device);
        s_device = NULL;
    }
    if (s_factory != NULL)
    {
        s_factory->lpVtbl->Release(s_factory);
        s_factory = NULL;
    }
}

int mainc(int argc, const char* argv[])
{
    bool useWARPAdapter = false;

    if (argc > 1)
    {
        if (strcmp(argv[1], "--warp") == 0)
        {
            puts("You have chosen the WARP adapter...");
            useWARPAdapter = true;
        }
    }

    do
    {
        if (!CreateD3D12Device(useWARPAdapter)) break;

        if (!CreateRootSignature()) break;

        if (!CreateComputePipelineStateObject()) break;

        if (!CreateComputeCommands()) break;

        if (!CreateBuffers()) break;

        if (!CreateFenceAndEvent()) break;

        HRESULT hRes = s_computeCommandList->lpVtbl->Close(s_computeCommandList);
        if (FAILED(hRes))
        {
            fprintf(stderr, "Execute init commands failed: %ld\n", hRes);
            break;
        }

        s_computeCommandQueue->lpVtbl->ExecuteCommandLists(s_computeCommandQueue, 1, (ID3D12CommandList* const []) { (ID3D12CommandList*)s_computeCommandList });

        SyncCommandQueue(s_computeCommandQueue, s_device, UPLOAD_DATA_SIGNAL_VALUE);

        // After finishing the whole buffer copy operation,
        // the intermediate buffer s_uploadBuffer can be released now.
        if (s_uploadBuffer != NULL)
        {
            s_uploadBuffer->lpVtbl->Release(s_uploadBuffer);
            s_uploadBuffer = NULL;
        }

        DoCompute();
    } while (false);

    ReleaseResources();
}
