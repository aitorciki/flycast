/*
    Created on: Sep 2, 2026

    This file is part of Flycast.
*/
#include "oit_profiler.h"

OITProfiler::~OITProfiler() = default;

#ifdef SWITCH_NVK_OIT_PROFILING
#include "../buffer.h"
#include "../commandpool.h"
#include "../vulkan_context.h"
#include "log/LogManager.h"

void OITProfiler::Init(CommandPool *commandPool)
{
	device = VulkanContext::Instance()->GetDevice();
	const auto queueProperties = VulkanContext::Instance()->GetPhysicalDevice().getQueueFamilyProperties();
	const u32 queueIndex = VulkanContext::Instance()->GetGraphicsQueueFamilyIndex();
	if (queueIndex >= queueProperties.size() || queueProperties[queueIndex].timestampValidBits == 0)
	{
		WARN_LOG(RENDERER, "OIT profiling disabled: graphics queue does not support timestamps");
		return;
	}
	timestampValidBits = queueProperties[queueIndex].timestampValidBits;
	timestampPeriod = VulkanContext::Instance()->GetPhysicalDevice().getProperties().limits.timestampPeriod;
	for (u32 i = 0; i < FlightCount; i++)
	{
		queryPools.emplace_back(device.createQueryPoolUnique(vk::QueryPoolCreateInfo({}, vk::QueryType::eTimestamp, MaxQueries)));
		counterReadbacks.emplace_back(std::make_unique<BufferData>(MaxPasses * sizeof(u32), vk::BufferUsageFlagBits::eTransferDst));
	}
	// Device creation enables all supported core features, so pipeline statistics
	// queries can be used whenever the feature is reported here.
	if (VulkanContext::Instance()->GetPhysicalDevice().getFeatures().pipelineStatisticsQuery)
	{
		const vk::QueryPipelineStatisticFlags statFlags = vk::QueryPipelineStatisticFlagBits::eVertexShaderInvocations
				| vk::QueryPipelineStatisticFlagBits::eFragmentShaderInvocations;
		for (u32 i = 0; i < FlightCount; i++)
			statsPools.emplace_back(device.createQueryPoolUnique(vk::QueryPoolCreateInfo({}, vk::QueryType::ePipelineStatistics, 1, statFlags)));
		NOTICE_LOG(RENDERER, "OIT profiling: pipeline statistics queries enabled");
	}
	else
		NOTICE_LOG(RENDERER, "OIT profiling: pipeline statistics queries not supported");
	frames.resize(FlightCount);
	commandPool->SetFrameCompleteCallback([this](int flightIndex) { OnFrameComplete(flightIndex); });
	NOTICE_LOG(RENDERER, "OIT profiling enabled: max %u render passes/frame, timestampValidBits=%u, period=%.3f ns/tick (NVK timestamp period is unverified)",
			MaxPasses, timestampValidBits, timestampPeriod);
}

void OITProfiler::Term()
{
	statsPools.clear();
	counterReadbacks.clear();
	counterStaging.clear();
	queryPools.clear();
	frames.clear();
	device = nullptr;
	activeFlight = -1;
}

void OITProfiler::Begin(vk::CommandBuffer commandBuffer, int flightIndex)
{
	if (queryPools.empty())
		return;
	if (activeFlight == flightIndex)
		return;
	if (activeFlight != -1 && activeFlight < (int)frames.size())
		// The previous frame never ended (e.g. aborted draw): drop its data
		frames[activeFlight].recording = false;
	if (flightIndex < 0 || flightIndex >= (int)frames.size())
	{
		activeFlight = -1;
		return;
	}
	activeFlight = flightIndex;
	FrameData& frame = frames[flightIndex];
	frame = FrameData{};
	frame.queryCount = 1;
	frame.recording = true;
	commandBuffer.resetQueryPool(*queryPools[flightIndex], 0, MaxQueries);
	commandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *queryPools[flightIndex], 0);
	if (!statsPools.empty())
	{
		commandBuffer.resetQueryPool(*statsPools[flightIndex], 0, 1);
		commandBuffer.beginQuery(*statsPools[flightIndex], 0, vk::QueryControlFlags());
	}
}

void OITProfiler::BeginPass(vk::CommandBuffer commandBuffer, vk::Buffer pixelCounter, vk::DeviceSize counterBytes)
{
	if (activeFlight == -1)
		return;
	FrameData& frame = frames[activeFlight];
#ifndef OIT_KBUFFER
	// Make the pixel-counter reset (transfer write) visible to the capture shaders.
	// In K-buffer mode OITBuffers::ResetPixelCounter already issues this barrier.
	vk::BufferMemoryBarrier counterBarrier(vk::AccessFlagBits::eTransferWrite,
			vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
			VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
			pixelCounter, 0, counterBytes > 0 ? counterBytes : sizeof(u32));
	commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader,
			vk::DependencyFlagBits::eByRegion, nullptr, counterBarrier, nullptr);
#endif
	// Keep the last query slot reserved for the frame-end timestamp
	if (frame.queryCount + 2 > MaxQueries - 1)
	{
		frame.truncatedPassCount++;
		frame.currentPassSkipped = true;
		return;
	}
	commandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *queryPools[activeFlight], frame.queryCount++);
}

void OITProfiler::EndPass(vk::CommandBuffer commandBuffer, vk::Buffer pixelCounter, vk::DeviceSize counterBytes, u32 pixelCapacity, bool autosort)
{
	if (activeFlight == -1)
		return;
	FrameData& frame = frames[activeFlight];
	if (frame.currentPassSkipped)
	{
		frame.currentPassSkipped = false;
		return;
	}
	commandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *queryPools[activeFlight], frame.queryCount++);
	if (counterBytes > 0 && frame.counterBytes == 0)
	{
		frame.counterBytes = counterBytes;
		frame.pixelCounter = pixelCounter;
	}
	if (counterBytes <= sizeof(u32))
	{
		// Legacy mode: copy the 4-byte atomic counter after the render pass.
		// The host read happens once this flight's fence has signaled (OnFrameComplete).
		if (pixelCounter && frame.passCount < MaxPasses)
		{
			vk::BufferMemoryBarrier fragToTransfer(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead,
					VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, pixelCounter, 0, sizeof(u32));
			commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader, vk::PipelineStageFlagBits::eTransfer,
					vk::DependencyFlagBits::eByRegion, nullptr, fragToTransfer, nullptr);
			const u32 slot = frame.passCount;
			commandBuffer.copyBuffer(pixelCounter, *counterReadbacks[activeFlight]->buffer,
					vk::BufferCopy(0, slot * sizeof(u32), sizeof(u32)));
			vk::BufferMemoryBarrier transferToHost(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead,
					VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, *counterReadbacks[activeFlight]->buffer,
					slot * sizeof(u32), sizeof(u32));
			commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
					vk::DependencyFlagBits::eByRegion, nullptr, transferToHost, nullptr);
			// Order the counter read above before the next pass's counter reset (transfer write)
			vk::BufferMemoryBarrier readBeforeReset(vk::AccessFlagBits::eTransferRead, vk::AccessFlagBits::eTransferWrite,
					VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, pixelCounter, 0, sizeof(u32));
			commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
					vk::DependencyFlagBits::eByRegion, nullptr, readBeforeReset, nullptr);
			frame.passes[frame.passCount] = PassData{ pixelCapacity, autosort };
		}
	}
	else if (pixelCounter && frame.passCount < MaxPasses)
	{
		// K-buffer mode: the whole per-pixel counter buffer is copied to staging in End()
		frame.passes[frame.passCount] = PassData{ pixelCapacity, autosort };
	}
	frame.passCount++;
}

void OITProfiler::CpuBegin(CpuPhase phase)
{
	if (activeFlight == -1)
		return;
	FrameData& frame = frames[activeFlight];
	frame.cpuStarts[(size_t)phase] = std::chrono::steady_clock::now();
	frame.cpuActive[(size_t)phase] = true;
}

void OITProfiler::CpuEnd(CpuPhase phase)
{
	if (activeFlight == -1)
		return;
	FrameData& frame = frames[activeFlight];
	if (!frame.cpuActive[(size_t)phase])
		return;
	frame.cpuActive[(size_t)phase] = false;
	frame.cpuNanos[(size_t)phase] += std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::steady_clock::now() - frame.cpuStarts[(size_t)phase]).count();
}

void OITProfiler::End(vk::CommandBuffer commandBuffer)
{
	if (activeFlight == -1)
		return;
	FrameData& frame = frames[activeFlight];
	commandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *queryPools[activeFlight], MaxQueries - 1);
	if (!statsPools.empty())
		commandBuffer.endQuery(*statsPools[activeFlight], 0);
	if (frame.counterBytes > sizeof(u32) && frame.pixelCounter
			&& activeFlight >= 0 && activeFlight < (int)frames.size())
	{
		// K-buffer mode: copy the whole per-pixel counter buffer to per-flight
		// staging. The host read happens once this flight's fence has signaled.
		if (activeFlight >= (int)counterStaging.size())
			counterStaging.resize(activeFlight + 1);
		auto& staging = counterStaging[activeFlight];
		// The previous submission for this flight slot has completed (its fence
		// was waited before this frame was recorded), so reallocating is safe.
		if (!staging || staging->bufferSize < frame.counterBytes)
			staging = std::make_unique<BufferData>(frame.counterBytes, vk::BufferUsageFlagBits::eTransferDst);
		vk::BufferMemoryBarrier fragToTransfer(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead,
				VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, frame.pixelCounter, 0, frame.counterBytes);
		commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader, vk::PipelineStageFlagBits::eTransfer,
				vk::DependencyFlagBits::eByRegion, nullptr, fragToTransfer, nullptr);
		commandBuffer.copyBuffer(frame.pixelCounter, *staging->buffer, vk::BufferCopy(0, 0, frame.counterBytes));
		vk::BufferMemoryBarrier transferToHost(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead,
				VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, *staging->buffer, 0, frame.counterBytes);
		commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
				vk::DependencyFlagBits::eByRegion, nullptr, transferToHost, nullptr);
		// Order the read above before the next frame's counter reset (transfer write)
		vk::BufferMemoryBarrier readBeforeReset(vk::AccessFlagBits::eTransferRead, vk::AccessFlagBits::eTransferWrite,
				VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, frame.pixelCounter, 0, frame.counterBytes);
		commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
				vk::DependencyFlagBits::eByRegion, nullptr, readBeforeReset, nullptr);
	}
	frame.recording = false;
	activeFlight = -1;
}

u64 OITProfiler::Delta(u64 start, u64 end, u32 validBits)
{
	if (validBits == 64)
		return end - start;
	const u64 mask = (1ULL << validBits) - 1;
	return (end - start) & mask;
}

void OITProfiler::OnFrameComplete(int flightIndex)
{
	if (flightIndex < 0 || flightIndex >= (int)frames.size())
		return;
	FrameData& frame = frames[flightIndex];
	if (frame.recording || frame.queryCount == 0)
		return;

	std::array<u64, MaxQueries> values{};
	vk::Result result = device.getQueryPoolResults(*queryPools[flightIndex], 0, frame.queryCount,
			frame.queryCount * sizeof(u64), values.data(), sizeof(u64), vk::QueryResultFlagBits::e64);
	if (result == vk::Result::eSuccess)
		result = device.getQueryPoolResults(*queryPools[flightIndex], MaxQueries - 1, 1,
				sizeof(u64), &values[MaxQueries - 1], sizeof(u64), vk::QueryResultFlagBits::e64);
	if (result != vk::Result::eSuccess)
	{
		WARN_LOG(RENDERER, "OIT profiling query results unavailable: %d", (int)result);
		return;
	}

	u64 frameTotal = Delta(values[0], values[MaxQueries - 1], timestampValidBits);
	u64 framePassTicks = 0;
	u64 frameMaxPass = 0;
	if (frame.passCount > 0)
	{
		for (u32 p = 0; p < frame.passCount; p++)
		{
			const u64 ticks = Delta(values[1 + 2 * p], values[2 + 2 * p], timestampValidBits);
			framePassTicks += ticks;
			if (ticks > frameMaxPass)
				frameMaxPass = ticks;
		}
	}
	totalTicks += frameTotal;
	passTicks += framePassTicks;
	if (frameMaxPass > maxPassTicks)
		maxPassTicks = frameMaxPass;
	truncatedPasses += frame.truncatedPassCount;
	if (frame.counterBytes > sizeof(u32))
	{
		// K-buffer mode: the staging buffer holds the per-pixel counters of the
		// last render pass of the frame. Summing hundreds of thousands of words
		// is too expensive per frame on the Switch CPU, so only sample 1 in 8.
		if (sampleCount % 8 == 0 && flightIndex < (int)counterStaging.size()
				&& counterStaging[flightIndex] && counterStaging[flightIndex]->bufferSize >= frame.counterBytes)
		{
			std::vector<u32> counters((u32)(frame.counterBytes / sizeof(u32)));
			counterStaging[flightIndex]->download((u32)frame.counterBytes, counters.data());
			const u32 slots = frame.passCount > 0 ? frame.passes[frame.passCount - 1].pixelCapacity : 0;
			u64 attempts = 0;
			u32 peakPixel = 0;
			u64 saturated = 0;
			for (u32 v : counters)
			{
				attempts += v;
				if (v > peakPixel)
					peakPixel = v;
				if (slots > 0 && v > slots)
					saturated++;
			}
			kbufferAttempts += attempts;
			kbufferSaturated += saturated;
			if (peakPixel > kbufferPeakPixel)
				kbufferPeakPixel = peakPixel;
			kbufferSamples++;
		}
	}
	else if (frame.passCount > 0)
	{
		u64 frameAttempts = 0;
		u64 frameOverflow = 0;
		u32 frameOverflowPasses = 0;
		std::array<u32, MaxPasses> counters{};
		counterReadbacks[flightIndex]->download((u32)(frame.passCount * sizeof(u32)), counters.data());
		for (u32 p = 0; p < frame.passCount; p++)
		{
			const u32 attempts = counters[p];
			frameAttempts += attempts;
			const u64 overflow = attempts > frame.passes[p].pixelCapacity ? (u64)attempts - frame.passes[p].pixelCapacity : 0;
			frameOverflow += overflow;
			if (overflow > 0)
				frameOverflowPasses++;
			if (attempts > peakAttempts)
				peakAttempts = attempts;
		}
		totalAttempts += frameAttempts;
		overflowAttempts += frameOverflow;
		overflowPasses += frameOverflowPasses;
	}
	for (size_t i = 0; i < CpuPhaseCount; i++)
		cpuNanos[i] += frame.cpuNanos[i];
	if (!statsPools.empty())
	{
		std::array<u64, 2> stats{};
		if (device.getQueryPoolResults(*statsPools[flightIndex], 0, 1, stats.size() * sizeof(u64),
					stats.data(), sizeof(u64), vk::QueryResultFlagBits::e64) == vk::Result::eSuccess)
		{
			totalVertInvocations += stats[0];
			totalFragInvocations += stats[1];
		}
	}
	if (++sampleCount != 120)
		return;

	const double avgTotalTicks = (double)totalTicks / sampleCount;
	const double avgPassTicks = (double)passTicks / sampleCount;
	NOTICE_LOG(RENDERER, "OIT GPU avg (%u frames): total=%.0f ticks (~%.2f ms), render-pass sum=%.0f ticks, max pass=%llu ticks (NVK timestamp period is unverified)",
			sampleCount, avgTotalTicks, avgTotalTicks * timestampPeriod / 1000000.0, avgPassTicks,
			(unsigned long long)maxPassTicks);
	NOTICE_LOG(RENDERER, "OIT CPU avg (%u frames) ns: reset=%llu depth=%llu opaque=%llu clear=%llu capture=%llu resolve=%llu continuation=%llu",
			sampleCount,
			(unsigned long long)(cpuNanos[0] / sampleCount),
			(unsigned long long)(cpuNanos[1] / sampleCount),
			(unsigned long long)(cpuNanos[2] / sampleCount),
			(unsigned long long)(cpuNanos[3] / sampleCount),
			(unsigned long long)(cpuNanos[4] / sampleCount),
			(unsigned long long)(cpuNanos[5] / sampleCount),
			(unsigned long long)(cpuNanos[6] / sampleCount));
	if (kbufferSamples > 0)
		NOTICE_LOG(RENDERER, "OIT per-pixel counters avg (%u frames, %u sampled): attempts=%llu/frame, peak pixel=%u slots, saturated pixels=%llu/frame",
				sampleCount, kbufferSamples, (unsigned long long)(kbufferAttempts / kbufferSamples),
				kbufferPeakPixel, (unsigned long long)(kbufferSaturated / kbufferSamples));
	else
		NOTICE_LOG(RENDERER, "OIT atomics avg (%u frames): attempts=%llu/frame, peak pass=%llu, overflow=%llu/frame on %u passes, truncated passes=%u",
				sampleCount, (unsigned long long)(totalAttempts / sampleCount), (unsigned long long)peakAttempts,
				(unsigned long long)(overflowAttempts / sampleCount), overflowPasses, truncatedPasses);
	if (!statsPools.empty())
		NOTICE_LOG(RENDERER, "OIT pipeline stats avg (%u frames): vertex invocations=%llu fragment invocations=%llu",
				sampleCount, (unsigned long long)(totalVertInvocations / sampleCount),
				(unsigned long long)(totalFragInvocations / sampleCount));
	totalTicks = 0;
	passTicks = 0;
	maxPassTicks = 0;
	totalAttempts = 0;
	peakAttempts = 0;
	overflowAttempts = 0;
	totalVertInvocations = 0;
	totalFragInvocations = 0;
	kbufferAttempts = 0;
	kbufferSaturated = 0;
	kbufferPeakPixel = 0;
	kbufferSamples = 0;
	cpuNanos.fill(0);
	overflowPasses = 0;
	truncatedPasses = 0;
	sampleCount = 0;
}
#endif
