/*
    Created on: Sep 2, 2026

    This file is part of Flycast.
*/
#pragma once

#include "../vulkan.h"

class CommandPool;
struct BufferData;

#ifdef SWITCH_NVK_OIT_PROFILING
#include <array>
#include <chrono>
#include <memory>
#include <vector>
#endif

class OITProfiler
{
public:
	enum class CpuPhase { Reset, Depth, Opaque, Clear, Capture, Resolve, Continuation, Count };

	~OITProfiler();
#ifdef SWITCH_NVK_OIT_PROFILING
	void Init(CommandPool *commandPool);
	void Term();
	// Called once per frame, before the first render pass
	void Begin(vk::CommandBuffer commandBuffer, int flightIndex);
	// Called after the pixel-counter reset and before beginRenderPass
	void BeginPass(vk::CommandBuffer commandBuffer, vk::Buffer pixelCounter, vk::DeviceSize counterBytes);
	// Called after endRenderPass; reads back the atomic counter for this pass
	void EndPass(vk::CommandBuffer commandBuffer, vk::Buffer pixelCounter, vk::DeviceSize counterBytes, u32 pixelCapacity, bool autosort);
	void CpuBegin(CpuPhase phase);
	void CpuEnd(CpuPhase phase);
	// Called after the last render pass, before the command buffer is ended
	void End(vk::CommandBuffer commandBuffer);

private:
	static constexpr u32 FlightCount = 2;
	static constexpr u32 MaxPasses = 16;
	// Frame start + 2 queries per pass + reserved frame end
	static constexpr u32 MaxQueries = MaxPasses * 2 + 2;
	static constexpr size_t CpuPhaseCount = (size_t)CpuPhase::Count;
	struct PassData {
		u32 pixelCapacity = 0;
		bool autosort = false;
	};
	struct FrameData {
		u32 queryCount = 0;
		u32 passCount = 0;
		u32 truncatedPassCount = 0;
		bool currentPassSkipped = false;
		// K-buffer mode: size of the per-pixel counter buffer and the buffer handle
		vk::DeviceSize counterBytes = 0;
		vk::Buffer pixelCounter;
		std::array<PassData, MaxPasses> passes{};
		std::array<u64, CpuPhaseCount> cpuNanos{};
		std::array<std::chrono::steady_clock::time_point, CpuPhaseCount> cpuStarts{};
		std::array<bool, CpuPhaseCount> cpuActive{};
		bool recording = false;
	};

	void OnFrameComplete(int flightIndex);
	static u64 Delta(u64 start, u64 end, u32 validBits);

	vk::Device device;
	std::vector<vk::UniqueQueryPool> queryPools;
	std::vector<vk::UniqueQueryPool> statsPools;
	std::vector<std::unique_ptr<BufferData>> counterReadbacks;
	// K-buffer mode: whole per-pixel counter buffer staging, one per flight
	std::vector<std::unique_ptr<BufferData>> counterStaging;
	std::vector<FrameData> frames;
	int activeFlight = -1;
	u32 timestampValidBits = 0;
	float timestampPeriod = 0.f;
	u64 totalTicks = 0;
	u64 passTicks = 0;
	u64 maxPassTicks = 0;
	u64 totalAttempts = 0;
	u64 peakAttempts = 0;
	u64 overflowAttempts = 0;
	u64 totalVertInvocations = 0;
	u64 totalFragInvocations = 0;
	// K-buffer aggregation (sampled)
	u64 kbufferAttempts = 0;
	u64 kbufferSaturated = 0;
	u32 kbufferPeakPixel = 0;
	u32 kbufferSamples = 0;
	std::array<u64, CpuPhaseCount> cpuNanos{};
	u32 overflowPasses = 0;
	u32 truncatedPasses = 0;
	u32 sampleCount = 0;
#else
	void Init(CommandPool *) {}
	void Term() {}
	void Begin(vk::CommandBuffer, int) {}
	void BeginPass(vk::CommandBuffer, vk::Buffer, vk::DeviceSize) {}
	void EndPass(vk::CommandBuffer, vk::Buffer, vk::DeviceSize, u32, bool) {}
	void CpuBegin(CpuPhase) {}
	void CpuEnd(CpuPhase) {}
	void End(vk::CommandBuffer) {}
#endif
};
