#include <fstream>
#include <iostream>
#include <locale> 
#include <codecvt> 

#include "utils.h"
#include "log.h"
#include "fgac_dx12_api.h"
#include "d3dx12.h"

#if ENABLE_RDC_FRAME_CAPTURE
#include "renderdoc_app.h"
#endif

#if ENABLE_PIX_FRAME_CAPTURE
#include "pix3.h"
#endif

#pragma comment(lib,"d3dcompiler.lib")
#pragma comment(lib,"dxcompiler.lib")
#pragma comment(lib,"D3D12.lib")
#pragma comment(lib,"dxgi.lib")

#include "stb_image.h"

//  TODO:
//  1.CommondQueue Compute/Direct/Transfer
//  2.Copy Dest to Common
//  3.use root const instead of constant buffer
//  4. texture mip generation
//  5. static sampler

#define MAX_HARDWARE_ADAPTER_COUNT 16

using wstring_conveter = std::wstring_convert<std::codecvt_utf8<wchar_t>>;

static void ThrowIfFailed(HRESULT hr) { if (FAILED(hr)) { assert(false); } }

static const D3D12_HEAP_PROPERTIES defaultHeapProperies =
{
    D3D12_HEAP_TYPE_DEFAULT,
    D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
    D3D12_MEMORY_POOL_UNKNOWN,
    1,1
};

static const D3D12_HEAP_PROPERTIES uploadHeapProperies =
{
    D3D12_HEAP_TYPE_UPLOAD,
    D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
    D3D12_MEMORY_POOL_UNKNOWN,
    1,1
};

static const D3D12_HEAP_PROPERTIES readBackHeapProperies =
{
    D3D12_HEAP_TYPE_READBACK,
    D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
    D3D12_MEMORY_POOL_UNKNOWN,
    1,1
};

template<class BlotType>
std::string ConvertBlobToString(BlotType* pBlob)
{
    std::vector<char> infoLog(pBlob->GetBufferSize() + 1);
    memcpy(infoLog.data(), pBlob->GetBufferPointer(), pBlob->GetBufferSize());
    infoLog[pBlob->GetBufferSize()] = 0;
    return std::string(infoLog.data());
}

static ID3D12DescriptorHeapPtr DxCreateDescriptorHeap(ID3D12Device5Ptr pDevice, uint32_t count, D3D12_DESCRIPTOR_HEAP_TYPE type, bool shaderVisible)
{
    D3D12_DESCRIPTOR_HEAP_DESC desc = {};
    desc.NumDescriptors = count;
    desc.Type = type;
    desc.Flags = shaderVisible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

    ID3D12DescriptorHeapPtr pHeap;
    ThrowIfFailed(pDevice->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&pHeap)));
    return pHeap;
}

#if ENABLE_RDC_FRAME_CAPTURE
static RENDERDOC_API_1_6_0* GetRenderDocApi()
{
    RENDERDOC_API_1_6_0* rdoc = nullptr;
    HMODULE module = LoadLibraryW(wstring_conveter().from_bytes(std::string(FGAC_ROOT_DIR) + "/thirdparty/renderdoc/renderdoc.dll").c_str());

    if (module == NULL)
    {
        return nullptr;
}

    pRENDERDOC_GetAPI getApi = nullptr;
    getApi = (pRENDERDOC_GetAPI)GetProcAddress(module, "RENDERDOC_GetAPI");

    if (getApi == nullptr)
    {
        return nullptr;
    }

    if (getApi(eRENDERDOC_API_Version_1_6_0, (void**)&rdoc) != 1)
    {
        return nullptr;
    }

    return rdoc;
}
#endif

void CDxDevice::InitDevice()
{
#if ENABLE_RDC_FRAME_CAPTURE
    rdoc = GetRenderDocApi();
#endif

#if ENABLE_PIX_FRAME_CAPTURE
    m_pixModule = PIXLoadLatestWinPixGpuCapturerLibrary();
#endif

#if defined(DEBUG) || defined(_DEBUG)
    ID3D12DebugPtr pDx12Debug;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&pDx12Debug)))) { pDx12Debug->EnableDebugLayer(); }
#endif

    IDXGIFactory4Ptr pDxgiFactory;
    ThrowIfFailed(CreateDXGIFactory1(IID_PPV_ARGS(&pDxgiFactory)));

    IDXGIAdapter1Ptr pAdapters[MAX_HARDWARE_ADAPTER_COUNT];
    
    UINT foundAdapterCount;
    for (foundAdapterCount = 0; foundAdapterCount < MAX_HARDWARE_ADAPTER_COUNT && pDxgiFactory->EnumAdapters1(foundAdapterCount, &pAdapters[foundAdapterCount]) != DXGI_ERROR_NOT_FOUND; ++foundAdapterCount) {}

    //0x10DE : Nvidia | 0x8086 : Intel | 0x1002 : AMD
    const int32_t vendorIndices[3] = { 0x10DE ,0x1002 ,0x8086 };
    int32_t vendorIndicesFound[3] = { -1 ,-1 ,-1 };

    for (int32_t index = 0; index < foundAdapterCount; index++)
    {
        DXGI_ADAPTER_DESC AdapterDesc;
        ThrowIfFailed(pAdapters[index]->GetDesc(&AdapterDesc));
        for (uint32_t vendorIndex = 0; vendorIndex < 3; vendorIndex++)
        {
            if (vendorIndices[vendorIndex] == AdapterDesc.VendorId) { vendorIndicesFound[vendorIndex] = index; }
        }
    }

    // create device
    for (uint32_t vendorIndex = 0; vendorIndex < 3; vendorIndex++)
    {
        if (vendorIndicesFound[vendorIndex] != -1)
        { 
            ThrowIfFailed(D3D12CreateDevice(pAdapters[vendorIndex], D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&m_pDevice)));

            DXGI_ADAPTER_DESC1 adapterDesc = { 0 };
            pAdapters[vendorIndicesFound[vendorIndex]]->GetDesc1(&adapterDesc);
            FGAC_LOG_INFO(std::format("Vendor ID {:X}", vendorIndices[vendorIndex]));
            FGAC_LOG_INFO(std::format("Dedicated Video Memory: {:.2f} GB", adapterDesc.DedicatedVideoMemory / (1024.0 * 1024.0 * 1024.0)));
            FGAC_LOG_INFO(std::format("Dedicated System Memory: {:.2f} GB", adapterDesc.DedicatedSystemMemory / (1024.0 * 1024.0 * 1024.0)));
            FGAC_LOG_INFO(std::format("Shared System Memory: {:.2f} GB", adapterDesc.SharedSystemMemory / (1024.0 * 1024.0 * 1024.0)));
        }
    }

    // create dxc compiler
    ThrowIfFailed(DxcCreateInstance(CLSID_DxcValidator, IID_PPV_ARGS(&m_dxcValidator)));
    ThrowIfFailed(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&m_pDxcCompiler)));
    ThrowIfFailed(DxcCreateInstance(CLSID_DxcLibrary, IID_PPV_ARGS(&m_pLibrary)));

    // create compute command queue
    D3D12_COMMAND_QUEUE_DESC queueDesc = { D3D12_COMMAND_LIST_TYPE_COMPUTE, 0, D3D12_COMMAND_QUEUE_FLAG_NONE };
    ThrowIfFailed(m_pDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_pCmpCmdQueue)));
    ThrowIfFailed(m_pDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&m_pCmdAllocator)));
    ThrowIfFailed(m_pDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, m_pCmdAllocator, nullptr, IID_PPV_ARGS(&m_pCmpCmdList)));

    // fence        
    ThrowIfFailed(m_pDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_pFence)));
    m_FenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    // 
    m_globalDescMan.Init(m_pDevice, 512, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, false);

    //
    shaderVisibleDescHeap = DxCreateDescriptorHeap(m_pDevice, 512, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, true);

#if ENABLE_RDC_FRAME_CAPTURE
    ((RENDERDOC_API_1_6_0*)rdoc)->StartFrameCapture(nullptr, nullptr);
#endif

#if ENABLE_PIX_FRAME_CAPTURE
    std::wstring pixPath = std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(FGAC_ROOT_DIR) + L"/build/pix.wpix";
    PIXCaptureParameters pixCaptureParameters;
    pixCaptureParameters.GpuCaptureParameters.FileName = pixPath.c_str();
    PIXBeginCapture(PIX_CAPTURE_GPU, &pixCaptureParameters);
#endif
}

static ID3D12RootSignaturePtr CreateRootSignature(ID3D12Device5Ptr pDevice, const D3D12_ROOT_SIGNATURE_DESC1& desc)
{
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC versionedRootDesc;
    versionedRootDesc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    versionedRootDesc.Desc_1_1 = desc;

    ID3DBlobPtr pSigBlob;
    ID3DBlobPtr pErrorBlob;
    ThrowIfFailed(D3D12SerializeVersionedRootSignature(&versionedRootDesc, &pSigBlob, &pErrorBlob));
    ID3D12RootSignaturePtr pRootSig;
    ThrowIfFailed(pDevice->CreateRootSignature(0, pSigBlob->GetBufferPointer(), pSigBlob->GetBufferSize(), IID_PPV_ARGS(&pRootSig)));
    return pRootSig;
}

IDxcBlobPtr CDxDevice::Dx12CompileCsLibraryDXC(const std::wstring& shaderPath, LPCWSTR pEntryPoint, DxcDefine* dxcDefine, uint32_t defineCount)
{
    std::ifstream shaderFile(shaderPath);

    if (shaderFile.good() == false)
    {
        ThrowIfFailed(-1);
    }

    std::string shader;
    std::stringstream strStream;
    strStream << shaderFile.rdbuf();
    shader = strStream.str();

    IDxcBlobEncodingPtr pTextBlob;
    IDxcOperationResultPtr pResult;
    HRESULT resultCode;
    IDxcBlobPtr pBlob;
    IDxcOperationResultPtr pValidResult;

    ThrowIfFailed(m_pLibrary->CreateBlobWithEncodingFromPinned((LPBYTE)shader.c_str(), (uint32_t)shader.size(), 0, &pTextBlob));
    ThrowIfFailed(m_pDxcCompiler->Compile(pTextBlob, shaderPath.data(), pEntryPoint, L"cs_6_1", nullptr, 0, dxcDefine, defineCount, nullptr, &pResult));
    ThrowIfFailed(pResult->GetStatus(&resultCode));

    if (FAILED(resultCode))
    {
        IDxcBlobEncodingPtr pError;
        ThrowIfFailed(pResult->GetErrorBuffer(&pError));
        std::string msg = ConvertBlobToString(pError.GetInterfacePtr());
        std::cout << msg;
        ThrowIfFailed(-1);
    }

    ThrowIfFailed(pResult->GetResult(&pBlob));
    m_dxcValidator->Validate(pBlob, DxcValidatorFlags_InPlaceEdit, &pValidResult);

    HRESULT validateStatus;
    pValidResult->GetStatus(&validateStatus);
    if (FAILED(validateStatus))
    {
        ThrowIfFailed(-1);
    }

    return pBlob;
}

void CDxDevice::CreateComputePipeline(SShaderResources shaderResources, const std::wstring& shaderPath)
{
    // compute pipeline
    SShaderResources computeResources = shaderResources;

    D3D12_ROOT_PARAMETER1  rootParams;
    std::vector<D3D12_DESCRIPTOR_RANGE1> descRanges;

    descRanges.resize(3);

    for (uint32_t descRangeIndex = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; descRangeIndex <= D3D12_DESCRIPTOR_RANGE_TYPE_CBV; descRangeIndex++)
    {
        D3D12_DESCRIPTOR_RANGE1 descRange;
        descRange.BaseShaderRegister = 0;
        descRange.NumDescriptors = 1;
        descRange.RegisterSpace = 0;
        descRange.OffsetInDescriptorsFromTableStart = descRangeIndex;
        descRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE(descRangeIndex);
        descRange.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE;
        descRanges[descRangeIndex] = descRange;
    }

    rootParams.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParams.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rootParams.DescriptorTable.NumDescriptorRanges = 3;
    rootParams.DescriptorTable.pDescriptorRanges = descRanges.data();

    D3D12_ROOT_SIGNATURE_DESC1 rootSigDesc = {};
    rootSigDesc.NumParameters = 1;
    rootSigDesc.pParameters = &rootParams;
    rootSigDesc.NumStaticSamplers = 0;
    rootSigDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

    m_pCsGlobalRootSig = CreateRootSignature(m_pDevice, rootSigDesc);

    ID3DBlobPtr csShader = Dx12CompileCsLibraryDXC(shaderPath, L"MainCS", nullptr, 0);

    D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc = {};
    computePsoDesc.pRootSignature = m_pCsGlobalRootSig;
    computePsoDesc.CS.pShaderBytecode = csShader->GetBufferPointer();
    computePsoDesc.CS.BytecodeLength = csShader->GetBufferSize();

    ThrowIfFailed(m_pDevice->CreateComputePipelineState(&computePsoDesc, IID_PPV_ARGS(&m_pCsPipelinState)));

}


ID3D12ResourcePtr CDxDevice::CreateDefaultBuffer(const void* pInitData, UINT64 nByteSize, ID3D12ResourcePtr& pUploadBuffer)
{
    ID3D12ResourcePtr defaultBuffer;

    D3D12_RESOURCE_DESC bufferDesc =
    {
        D3D12_RESOURCE_DIMENSION_BUFFER,
        0,nByteSize, 1,1,1,
        DXGI_FORMAT_UNKNOWN,
        1, 0,
        D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        D3D12_RESOURCE_FLAG_NONE
    };

    ThrowIfFailed(m_pDevice->CreateCommittedResource(&defaultHeapProperies, D3D12_HEAP_FLAG_NONE, &bufferDesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&defaultBuffer)));
    ThrowIfFailed(m_pDevice->CreateCommittedResource(&uploadHeapProperies, D3D12_HEAP_FLAG_NONE, &bufferDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&pUploadBuffer)));


    D3D12_RESOURCE_BARRIER barrierBefore = {};
    barrierBefore.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrierBefore.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrierBefore.Transition.pResource = defaultBuffer;
    barrierBefore.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barrierBefore.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    barrierBefore.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_pCmpCmdList->ResourceBarrier(1, &barrierBefore);

    {
        void* DestData = nullptr;
        D3D12_RANGE range = { 0,0 };
        pUploadBuffer->Map(0, &range, &DestData);
        memcpy(DestData, pInitData, nByteSize);
        pUploadBuffer->Unmap(0, nullptr);
        m_pCmpCmdList->CopyBufferRegion(defaultBuffer, 0, pUploadBuffer, 0, nByteSize);
    }

    D3D12_RESOURCE_BARRIER barrierAfter = {};
    barrierAfter.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrierAfter.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrierAfter.Transition.pResource = defaultBuffer;
    barrierAfter.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrierAfter.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    barrierAfter.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_pCmpCmdList->ResourceBarrier(1, &barrierAfter);

    return defaultBuffer;

}
#define TEST_BLOCK_SIZE 4
#define CS_GROUP_SIZE 8


STextureBatch CDxDevice::CreateTextureBatch()
{
    OpenCmdListImpl();

    // create texture reousrce begin
    STextureBatch texBatch;
    texBatch.m_batchTexNum = 1;
    texBatch.m_textureBlockSize = Vec2(TEST_BLOCK_SIZE, TEST_BLOCK_SIZE);

    std::string imagePath("G:/fgac/build/test.jpeg");
    int width = 0, height = 0, comp = 0;
    stbi_uc* data = stbi_load(imagePath.c_str(), &width, &height, &comp, STBI_rgb_alpha);
    texBatch.m_textureSize = Vec2i(width, height);
    
    {
        D3D12_RESOURCE_DESC resDesc = {};
        resDesc.DepthOrArraySize = 1;
        resDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        resDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        resDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
        resDesc.Width = width;
        resDesc.Height = height;
        resDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        resDesc.MipLevels = 1;
        resDesc.SampleDesc.Count = 1;

        ThrowIfFailed(m_pDevice->CreateCommittedResource(
            &defaultHeapProperies,
            D3D12_HEAP_FLAG_NONE,
            &resDesc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&m_pTestResource)));

        UINT64 requiredSize = 0;
        m_pDevice->GetCopyableFootprints(&resDesc, 0, 1, 0, nullptr, nullptr, nullptr, &requiredSize);

        D3D12_RESOURCE_DESC uploadResDesc = {};
        uploadResDesc.DepthOrArraySize = 1;
        uploadResDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        uploadResDesc.Format = DXGI_FORMAT_UNKNOWN;
        uploadResDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
        uploadResDesc.Width = requiredSize;
        uploadResDesc.Height = 1;
        uploadResDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        uploadResDesc.MipLevels = 1;
        uploadResDesc.SampleDesc.Count = 1;
        ThrowIfFailed(m_pDevice->CreateCommittedResource(
            &uploadHeapProperies,
            D3D12_HEAP_FLAG_NONE,
            &uploadResDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&m_pTestResourceUpload)));


        D3D12_SUBRESOURCE_DATA textureData = {};
        textureData.pData = data;
        textureData.RowPitch = width * sizeof(uint8_t) * 4;
        textureData.SlicePitch = textureData.RowPitch * height;

        CopyDataFromUploadToDefaulHeap(m_pCmpCmdList, m_pTestResource, m_pTestResourceUpload, 0, 0, 1, &textureData);

        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.Transition.pResource = m_pTestResource;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

        m_pCmpCmdList->ResourceBarrier(1, &barrier);

        //texBatch.m_reourceDescs
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Format = resDesc.Format;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MipLevels = 1;

        uint32_t srvIndex = m_globalDescMan.AllocDesc();
        D3D12_CPU_DESCRIPTOR_HANDLE srvHandle = m_globalDescMan.GetCPUHandle(srvIndex);
        m_pDevice->CreateShaderResourceView(m_pTestResource, &srvDesc, srvHandle);

        texBatch.m_reourceDescs.resize(1);
        texBatch.m_reourceDescs[0].srvDesc = srvHandle;
    }
    
    uint32_t totalBlockNum = 0;
    {
        uint32_t dimSize = TEST_BLOCK_SIZE;
        static constexpr uint32_t maxMipNum = 14;
        uint32_t totalMipsNum = 0;
        
        uint32_t mipStartBlockIndex[maxMipNum];
        mipStartBlockIndex[0] = 0;

        for (uint32_t mipIndex = 0; mipIndex < maxMipNum; mipIndex++)
        {
            uint32_t mipWidth = width >> mipIndex;
            uint32_t mipHeight = height >> mipIndex;

            uint32_t xBlockNum = (mipWidth + dimSize - 1) / dimSize;
            uint32_t yBlockNum = (mipHeight + dimSize - 1) / dimSize;

            totalBlockNum += (xBlockNum * yBlockNum);

            if (xBlockNum <= 0 || yBlockNum <= 0)
            {
                totalMipsNum++;
            }

            mipStartBlockIndex[mipIndex] = totalBlockNum;
        }

        uint32_t csGroupSize = CS_GROUP_SIZE * CS_GROUP_SIZE;
        uint32_t csGroupNum = (totalBlockNum + (csGroupSize - 1)) / csGroupSize;
        uint32_t csGroupNumX = (width + (dimSize - 1)) / dimSize;
        uint32_t csGroupNumY = (csGroupNum + (csGroupNumX - 1)) / csGroupNumX;

        texBatch.threadGroupCount = Vec2i(csGroupNumX, csGroupNumY);

        struct STexInfo
        {
            uint32_t mipStartBlockIndex[maxMipNum];
            uint32_t groupNumX;
            uint32_t m_blockNum;
            uint32_t m_mipsNum;
            uint32_t m_mip0TexWidth;
            uint32_t m_mip0TexHeight;
            Vec2i m_blockSize;

            uint8_t padding[255  - maxMipNum * sizeof(uint32_t) - sizeof(uint32_t) * 7];
        };

        static_assert(sizeof(STexInfo) == 256);

        STexInfo texInfo;
        memcpy(texInfo.mipStartBlockIndex, mipStartBlockIndex, sizeof(uint32_t)* maxMipNum);
        texInfo.groupNumX = csGroupNumX;
        texInfo.m_blockNum = totalBlockNum;
        texInfo.m_mipsNum = totalMipsNum;
        texInfo.m_mip0TexWidth = width;
        texInfo.m_mip0TexHeight = height;
        texInfo.m_blockSize = Vec2i(TEST_BLOCK_SIZE, TEST_BLOCK_SIZE);

        m_pTestCbReSource = CreateDefaultBuffer(&texInfo, sizeof(STexInfo), m_pTestCbReSourceUpload);
        D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc = {};
        cbvDesc.BufferLocation = m_pTestCbReSource->GetGPUVirtualAddress();
        cbvDesc.SizeInBytes = sizeof(STexInfo);

        uint32_t cbvIndex = m_globalDescMan.AllocDesc();
        D3D12_CPU_DESCRIPTOR_HANDLE cbvHandle = m_globalDescMan.GetCPUHandle(cbvIndex);
        m_pDevice->CreateConstantBufferView(&cbvDesc, cbvHandle);
        texBatch.m_reourceDescs[0].cbvDesc = cbvHandle;

    }

    // create uav
    {
        uint32_t outBufferSize = totalBlockNum * sizeof(uint32_t) * 4;

        D3D12_RESOURCE_DESC outResDesc = {};
        outResDesc.DepthOrArraySize = 1;
        outResDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        outResDesc.Format = DXGI_FORMAT_UNKNOWN;
        outResDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        outResDesc.Width = outBufferSize;
        outResDesc.Height = 1;
        outResDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        outResDesc.MipLevels = 1;
        outResDesc.SampleDesc.Count = 1;

        ThrowIfFailed(m_pDevice->CreateCommittedResource(
            &defaultHeapProperies,
            D3D12_HEAP_FLAG_NONE,
            &outResDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&m_pResourceOutUAV)));

        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = DXGI_FORMAT_UNKNOWN;
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavDesc.Buffer.FirstElement = 0;
        uavDesc.Buffer.NumElements = totalBlockNum;
        uavDesc.Buffer.StructureByteStride = sizeof(uint32_t) * 4;
        uavDesc.Buffer.CounterOffsetInBytes = 0;
        uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

        uint32_t uavIndex = m_globalDescMan.AllocDesc();
        D3D12_CPU_DESCRIPTOR_HANDLE outUavHandle = m_globalDescMan.GetCPUHandle(uavIndex);
        m_pDevice->CreateUnorderedAccessView(m_pResourceOutUAV, nullptr, &uavDesc, outUavHandle);
        texBatch.m_reourceDescs[0].uavDesc = outUavHandle;
    }

    CloseAndExecuteCmdListImpl();
    WaitGPUCmdListFinishImpl();
    OpenCmdListImpl();

    return texBatch;
}


void CDxDevice::CompressTexture(const STextureBatch& texBatch)
{
    //TODO: move this scope to init compress texture
    {
        m_pCmpCmdList->SetPipelineState(m_pCsPipelinState);
        m_pCmpCmdList->SetComputeRootSignature(m_pCsGlobalRootSig);
        ID3D12DescriptorHeap* ppHeaps[] = { shaderVisibleDescHeap };
        m_pCmpCmdList->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);
    }

    uint32_t nElemSize = m_pDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    
    uint32_t numCopy = 3;

    D3D12_CPU_DESCRIPTOR_HANDLE hCpuBegin = shaderVisibleDescHeap->GetCPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE hGpuBegin = shaderVisibleDescHeap->GetGPUDescriptorHandleForHeapStart();
    
    {
        D3D12_CPU_DESCRIPTOR_HANDLE m_viewHandles[3];
        m_viewHandles[0] = texBatch.m_reourceDescs[0].srvDesc;
        m_viewHandles[1] = texBatch.m_reourceDescs[0].uavDesc;
        m_viewHandles[2] = texBatch.m_reourceDescs[0].cbvDesc;
        m_pDevice->CopyDescriptors(1, &hCpuBegin, &numCopy, numCopy, m_viewHandles, nullptr, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }

    for (uint32_t index = 0; index < texBatch.m_batchTexNum; index++)
    {
        m_pCmpCmdList->SetComputeRootDescriptorTable(0, hGpuBegin); // cbv
        //m_pCmpCmdList->SetComputeRootDescriptorTable(1, D3D12_GPU_DESCRIPTOR_HANDLE{ hGpuBegin.ptr + static_cast<UINT64>(1 * nElemSize) }); // srv
        //m_pCmpCmdList->SetComputeRootDescriptorTable(2, D3D12_GPU_DESCRIPTOR_HANDLE{ hGpuBegin.ptr + static_cast<UINT64>(2 * nElemSize) }); // uav

        m_pCmpCmdList->Dispatch(texBatch.threadGroupCount.x, texBatch.threadGroupCount.y, 1);
    }

    {
        ID3D12CommandList* pComputeList = m_pCmpCmdList.GetInterfacePtr();

        m_pCmpCmdList->Close();
        m_pCmpCmdQueue->ExecuteCommandLists(1, &pComputeList);

        uint64_t signaledValue = m_nFenceValue;
        ThrowIfFailed(m_pCmpCmdQueue->Signal(m_pFence, signaledValue));
        m_nFenceValue++;

        if (m_pFence->GetCompletedValue() < signaledValue)
        {
            ThrowIfFailed(m_pFence->SetEventOnCompletion(signaledValue, m_FenceEvent));
            WaitForSingleObject(m_FenceEvent, INFINITE);
        }
    }

}

void CDxDevice::Shutdown()
{
#if ENABLE_PIX_FRAME_CAPTURE
    PIXEndCapture(false);
#endif

#if ENABLE_RDC_FRAME_CAPTURE
    ((RENDERDOC_API_1_6_0*)rdoc)->EndFrameCapture(nullptr, nullptr);
#endif
}

void CDxDevice::OpenCmdListImpl()
{
    if (m_cmdState == ECmdState::CS_CLOSE)
    {
        ThrowIfFailed(m_pCmpCmdList->Reset(m_pCmdAllocator, nullptr));
        m_cmdState = ECmdState::CS_OPENG;
    }
}

void CDxDevice::CloseAndExecuteCmdListImpl()
{
    if (m_cmdState == ECmdState::CS_OPENG)
    {
        m_cmdState = ECmdState::CS_CLOSE;
        ID3D12CommandList* pCmpCmdList = m_pCmpCmdList.GetInterfacePtr();

        m_pCmpCmdList->Close();
        m_pCmpCmdQueue->ExecuteCommandLists(1, &pCmpCmdList);
    }
}

void CDxDevice::WaitGPUCmdListFinishImpl()
{
    uint64_t mSignaledValue = m_nFenceValue;
    ThrowIfFailed(m_pCmpCmdQueue->Signal(m_pFence, mSignaledValue));
    m_nFenceValue++;

    if (m_pFence->GetCompletedValue() < mSignaledValue)
    {
        ThrowIfFailed(m_pFence->SetEventOnCompletion(mSignaledValue, m_FenceEvent));
        WaitForSingleObject(m_FenceEvent, INFINITE);
    }
}

void CDxDevice::ResetCmdAllocImpl()
{
    if (m_cmdState == ECmdState::CS_CLOSE)
    {
        ThrowIfFailed(m_pCmdAllocator->Reset());
    }
}

void Dx12CsTestFunc()
{
    SShaderResources shaderResources = {1,1,1,0,0};
    
    CDxDevice dxDevice;
    dxDevice.InitDevice();
    dxDevice.CreateComputePipeline(shaderResources, std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(FGAC_ROOT_DIR) + L"/include/shader/compressTex.hlsl");
    STextureBatch texBatch = dxDevice.CreateTextureBatch();
    dxDevice.CompressTexture(texBatch);
    dxDevice.Shutdown();
}


void CGlobalDescManager::Init(ID3D12Device5Ptr pDevice, uint32_t size, D3D12_DESCRIPTOR_HEAP_TYPE descHeapType, bool shaderVisible)
{
    m_pDevice = pDevice;
    m_pDescHeap = DxCreateDescriptorHeap(pDevice, size, descHeapType, shaderVisible);
    m_descHeapType = descHeapType;

    m_nextFreeDescIndex.resize(size);
    for (uint32_t index = 0; index < size; index++)
    {
        m_nextFreeDescIndex[index] = index + 1;
    }

    m_currFreeIndex = 0;
}

const uint32_t CGlobalDescManager::GetNumdDesc()
{
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc = m_pDescHeap->GetDesc();
    return heapDesc.NumDescriptors;
}

D3D12_CPU_DESCRIPTOR_HANDLE CGlobalDescManager::GetCPUHandle(uint32_t index)
{
    D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = m_pDescHeap->GetCPUDescriptorHandleForHeapStart();
    cpuHandle.ptr += m_pDevice->GetDescriptorHandleIncrementSize(m_descHeapType) * index;
    return cpuHandle;
}

D3D12_GPU_DESCRIPTOR_HANDLE CGlobalDescManager::GetGPUHandle(uint32_t index)
{
    D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle = m_pDescHeap->GetGPUDescriptorHandleForHeapStart();
    gpuHandle.ptr += m_pDevice->GetDescriptorHandleIncrementSize(m_descHeapType) * index;
    return gpuHandle;
}

ID3D12DescriptorHeapPtr CGlobalDescManager::GetHeap()
{
    return m_pDescHeap;
}

uint32_t CGlobalDescManager::AllocDesc()
{
    if (m_nextFreeDescIndex.size() <= m_currFreeIndex)
    {
        m_nextFreeDescIndex.resize((((m_currFreeIndex + 1) / 1024) + 1) * 1024);
        for (uint32_t index = m_currFreeIndex; index < m_nextFreeDescIndex.size(); index++)
        {
            m_nextFreeDescIndex[index] = index + 1;
        }
    }

    uint32_t allocIndex = m_currFreeIndex;
    m_currFreeIndex = m_nextFreeDescIndex[m_currFreeIndex];
    return allocIndex;
}

void CGlobalDescManager::FreeDesc(uint32_t freeIndex)
{
    m_nextFreeDescIndex[freeIndex] = m_currFreeIndex;
    m_currFreeIndex = freeIndex;
}
