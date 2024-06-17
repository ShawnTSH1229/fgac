#pragma once

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdalign.h>
#include <assert.h>
#include <string>
#include <vector>

//windows
#include <Windows.h>
#include <comdef.h>

//dx12
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <dxcapi.h>

#include "utils.h"

#define ENABLE_PIX_FRAME_CAPTURE 0
#define ENABLE_RDC_FRAME_CAPTURE 1


#define MAKE_SMART_COM_PTR(_a) _COM_SMARTPTR_TYPEDEF(_a, __uuidof(_a))
MAKE_SMART_COM_PTR(IDXGIFactory4);
MAKE_SMART_COM_PTR(ID3D12Device5);
MAKE_SMART_COM_PTR(IDXGIAdapter1);
MAKE_SMART_COM_PTR(ID3D12Debug);
MAKE_SMART_COM_PTR(ID3D12CommandQueue);
MAKE_SMART_COM_PTR(ID3D12CommandAllocator);
MAKE_SMART_COM_PTR(ID3D12GraphicsCommandList4);
MAKE_SMART_COM_PTR(ID3D12Resource);
MAKE_SMART_COM_PTR(ID3D12Fence);
MAKE_SMART_COM_PTR(IDxcBlobEncoding);
MAKE_SMART_COM_PTR(IDxcCompiler);
MAKE_SMART_COM_PTR(IDxcLibrary);
MAKE_SMART_COM_PTR(IDxcOperationResult);
MAKE_SMART_COM_PTR(IDxcBlob);
MAKE_SMART_COM_PTR(ID3DBlob);
MAKE_SMART_COM_PTR(ID3D12StateObject);
MAKE_SMART_COM_PTR(ID3D12RootSignature);
MAKE_SMART_COM_PTR(IDxcValidator);
MAKE_SMART_COM_PTR(ID3D12StateObjectProperties);
MAKE_SMART_COM_PTR(ID3D12DescriptorHeap);
MAKE_SMART_COM_PTR(ID3D12PipelineState);

struct SShaderResources
{
	uint32_t m_nSRV = 0;
	uint32_t m_nUAV = 0;
	uint32_t m_nCBV = 0;
	uint32_t m_nSampler = 0;
	uint32_t m_rootConstant = 0;

	uint32_t operator[](std::size_t index)
	{
		return *((uint32_t*)(this) + index);
	}
};

//**************************************
// TODO: Move these structure to common directory: BEGIN
//**************************************

struct SReourceDescs
{
	D3D12_CPU_DESCRIPTOR_HANDLE srvDesc; //input texture
	D3D12_CPU_DESCRIPTOR_HANDLE uavDesc; //output
	D3D12_CPU_DESCRIPTOR_HANDLE cbvDesc; //constant buffer
};

struct STextureBatch
{
	uint32_t m_batchTexNum;
	std::vector<SReourceDescs> m_reourceDescs;
	DXGI_FORMAT m_texFormat;

	Vec2 m_textureBlockSize;
	Vec2 m_textureSize;


	Vec2i threadGroupCount;
};

class CGlobalDescManager
{
public:
	CGlobalDescManager() {};

	void Init(ID3D12Device5Ptr pDevice, uint32_t size, D3D12_DESCRIPTOR_HEAP_TYPE descHeapType, bool shaderVisible);
	const uint32_t GetNumdDesc();
	D3D12_CPU_DESCRIPTOR_HANDLE GetCPUHandle(uint32_t index);
	D3D12_GPU_DESCRIPTOR_HANDLE GetGPUHandle(uint32_t index);
	ID3D12DescriptorHeapPtr GetHeap();
	uint32_t AllocDesc();
	void FreeDesc(uint32_t freeIndex);

private:
	ID3D12DescriptorHeapPtr m_pDescHeap;
	ID3D12Device5Ptr m_pDevice;

	D3D12_DESCRIPTOR_HEAP_TYPE m_descHeapType;
	std::vector<uint32_t>m_nextFreeDescIndex;
	uint32_t m_currFreeIndex;
};

enum class ECmdState
{
	CS_OPENG,
	CS_CLOSE
};

class CDxDevice
{
public:
	void InitDevice();
	void CreateComputePipeline(SShaderResources shaderResources, const std::wstring& shaderPath);
	STextureBatch CreateTextureBatch();
	void CompressTexture(const STextureBatch& texBatch);
	void Shutdown();
private:
	void OpenCmdListImpl();
	void CloseAndExecuteCmdListImpl();
	void WaitGPUCmdListFinishImpl();
	void ResetCmdAllocImpl();

	ID3D12ResourcePtr CreateDefaultBuffer(const void* pInitData, UINT64 nByteSize, ID3D12ResourcePtr& pUploadBuffer);

	IDxcBlobPtr Dx12CompileCsLibraryDXC(const std::wstring& shaderPath, LPCWSTR pEntryPoint, DxcDefine* dxcDefine, uint32_t defineCount);

	ID3D12Device5Ptr m_pDevice;

	ID3D12CommandQueuePtr m_pCmpCmdQueue;
	ID3D12CommandAllocatorPtr m_pCmdAllocator;
	ID3D12GraphicsCommandList4Ptr m_pCmpCmdList;

	// compute pso
	ID3D12RootSignaturePtr m_pCsGlobalRootSig;
	ID3D12PipelineStatePtr m_pCsPipelinState;

	// compile shader
	IDxcCompilerPtr m_pDxcCompiler;
	IDxcLibraryPtr m_pLibrary;
	IDxcValidatorPtr m_dxcValidator;

	// todo
	//ID3D12DescriptorHeapPtr gloablDescHeap;
	ID3D12DescriptorHeapPtr shaderVisibleDescHeap;

	// fence
	ID3D12FencePtr m_pFence;
	HANDLE m_FenceEvent;
	uint64_t m_nFenceValue = 0;;

	//temporary code
	CGlobalDescManager m_globalDescMan;
	ECmdState m_cmdState;
	ID3D12ResourcePtr m_pTestResource;
	ID3D12ResourcePtr m_pTestResourceUpload;
	ID3D12ResourcePtr m_pResourceOutUAV;

	ID3D12ResourcePtr m_pTestCbReSource;
	ID3D12ResourcePtr m_pTestCbReSourceUpload;

#if ENABLE_RDC_FRAME_CAPTURE
	void* rdoc;
#endif

#if ENABLE_PIX_FRAME_CAPTURE
	HMODULE m_pixModule;
#endif
};

void Dx12CsTestFunc();