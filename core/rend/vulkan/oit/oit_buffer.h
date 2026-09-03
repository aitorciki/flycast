/*
    Created on: Nov 12, 2019

	Copyright 2019 flyinghead

	This file is part of Flycast.

    Flycast is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 2 of the License, or
    (at your option) any later version.

    Flycast is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Flycast.  If not, see <https://www.gnu.org/licenses/>.
*/
#pragma once
#include "../buffer.h"
#include "../texture.h"

#include <algorithm>
#include <cinttypes>
#include <memory>

class OITBuffers
{
public:
	void Init(int width, int height)
	{
		const VulkanContext *context = VulkanContext::Instance();

		if (width <= maxWidth && height <= maxHeight)
			return;
		maxWidth = std::max(maxWidth, width);
		maxHeight = std::max(maxHeight, height);

		if (!pixelCounter)
		{
			pixelCounter = std::make_unique<BufferData>(4,
					vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst
					| vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eDeviceLocal);
			pixelCounterReset = std::make_unique<BufferData>(4, vk::BufferUsageFlagBits::eTransferSrc);
			const int zero = 0;
			pixelCounterReset->upload(sizeof(zero), &zero);
		}
		// We need to wait until this buffer is not used before deleting it
		context->WaitIdle();
		abufferPointer.reset();
		abufferPointer = std::make_unique<BufferData>(maxWidth * maxHeight * sizeof(int),
				vk::BufferUsageFlagBits::eStorageBuffer
#ifdef OIT_KBUFFER
					| vk::BufferUsageFlagBits::eTransferDst
#endif
				, vk::MemoryPropertyFlagBits::eDeviceLocal);
		firstFrameAfterInit = true;
#ifdef OIT_KBUFFER
		updatePixelSlots();
#endif
	}

	void updateDescriptorSet(vk::DescriptorSet descSet, std::vector<vk::WriteDescriptorSet>& writeDescSets)
	{
		static vk::DescriptorBufferInfo pixelBufferInfo({}, 0, vk::WholeSize);
		pixelBufferInfo.buffer = *pixelBuffer->buffer;
		/**
		 * The intention behind setting the range to 1 byte in BDA mode is to make it easier to spot any missing or invalid USE_BDA defines in shader sources.
		 * If USE_BDA is missing or invalid, the shader will access PixelBuffer as an SSBO, and if it's this tiny, it should result in obvious graphical glitches.
		 */
		pixelBufferInfo.range = pixelBufferAddress ? 1 : VK_WHOLE_SIZE;
		writeDescSets.emplace_back(descSet, 7, 0, vk::DescriptorType::eStorageBuffer, nullptr, pixelBufferInfo);
		static vk::DescriptorBufferInfo pixelCounterBufferInfo({}, 0, 4);
		pixelCounterBufferInfo.buffer = *pixelCounter->buffer;
		writeDescSets.emplace_back(descSet, 8, 0, vk::DescriptorType::eStorageBuffer, nullptr, pixelCounterBufferInfo);
		static vk::DescriptorBufferInfo abufferPointerInfo({}, 0, vk::WholeSize);
		abufferPointerInfo.buffer = *abufferPointer->buffer;
		writeDescSets.emplace_back(descSet, 9, 0, vk::DescriptorType::eStorageBuffer, nullptr, abufferPointerInfo);
	}

	void OnNewFrame(vk::CommandBuffer commandBuffer)
	{
		firstFrameAfterInit = false;
#ifdef OIT_KBUFFER
		const u32 oldSlots = pixelSlots;
		updatePixelSlots();
		// One Pixel node (16 bytes) per slot for every pixel
		const vk::DeviceSize requiredSize = (vk::DeviceSize)maxWidth * maxHeight * pixelSlots * 16;
		if (pixelBufferSize != config::PixelBufferSize || pixelSlots != oldSlots
				|| !pixelBuffer || pixelBuffer->bufferSize < requiredSize)
		{
			pixelBufferSize = config::PixelBufferSize;
			VulkanContext::Instance()->WaitIdle();
			makePixelBuffer();
		}
#else
		if (pixelBufferSize != config::PixelBufferSize)
		{
			pixelBufferSize = config::PixelBufferSize;
			VulkanContext::Instance()->WaitIdle();
			makePixelBuffer();
		}
#endif
	}

	void ResetPixelCounter(vk::CommandBuffer commandBuffer)
	{
#ifdef OIT_KBUFFER
		commandBuffer.fillBuffer(*abufferPointer->buffer, 0, VK_WHOLE_SIZE, 0);
		// Make the zeroed counters visible to the fragment shaders of the next render pass
		vk::BufferMemoryBarrier barrier(vk::AccessFlagBits::eTransferWrite,
				vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
				VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
				*abufferPointer->buffer, 0, VK_WHOLE_SIZE);
		commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader,
				vk::DependencyFlags(), {}, barrier, {});
#else
    	vk::BufferCopy copy(0, 0, sizeof(int));
    	commandBuffer.copyBuffer(*pixelCounterReset->buffer, *pixelCounter->buffer, copy);
#endif
	}

	void Term()
	{
		pixelBufferAddress = 0;
		pixelBuffer.reset();
		pixelCounter.reset();
		pixelCounterReset.reset();
		abufferPointer.reset();
		pixelSlots = 0;
	}

	bool isFirstFrameAfterInit() const { return firstFrameAfterInit; }

	vk::DeviceAddress getPixelBufferAddress() const { return pixelBufferAddress; }

	vk::Buffer getPixelCounter() const
	{
#ifdef OIT_KBUFFER
		return *abufferPointer->buffer;
#else
		return *pixelCounter->buffer;
#endif
	}

	// Number of per-pixel slots in the K-buffer. 0 when using the global-counter linked-list allocation.
	u32 getPixelSlots() const { return pixelSlots; }

	vk::DeviceSize getPixelCounterSize() const
	{
#ifdef OIT_KBUFFER
		return abufferPointer ? abufferPointer->bufferSize : 0;
#else
		return sizeof(int);
#endif
	}

	vk::DeviceSize getPixelBufferSize() const
	{
		return pixelBuffer ? pixelBuffer->bufferSize : 0;
	}

private:
	std::unique_ptr<BufferData> pixelBuffer;
	std::unique_ptr<BufferData> pixelCounter;
	std::unique_ptr<BufferData> pixelCounterReset;
	std::unique_ptr<BufferData> abufferPointer;
	bool firstFrameAfterInit = false;
	int maxWidth = 0;
	int maxHeight = 0;
	int64_t pixelBufferSize = 0;
	vk::DeviceAddress pixelBufferAddress = 0;
	u32 pixelSlots = 0;

#ifdef OIT_KBUFFER
	void updatePixelSlots()
	{
		// Number of per-pixel slots: bounded by the maximum per-pixel layer count and by the
		// maximum memory allocation size (one 16-byte node per slot and per pixel).
		u32 slots = std::min(std::max<u32>(config::PerPixelLayers, 1), 256u);
		const int64_t maxAllocSize = (int64_t)VulkanContext::Instance()->GetMaxMemoryAllocationSize();
		const int64_t pixelNodeSize = (int64_t)maxWidth * maxHeight * 16;
		while (slots > 1 && (int64_t)slots * pixelNodeSize > maxAllocSize)
			slots--;
		pixelSlots = slots;
	}
#endif

	void makePixelBuffer() {
		const VulkanContext *context = VulkanContext::Instance();
		const u32 maxStorageBufferRange = context->GetMaxStorageBufferRange();
#ifdef OIT_KBUFFER
		const vk::DeviceSize requiredSize = (vk::DeviceSize)maxWidth * maxHeight * pixelSlots * 16;
		vk::DeviceSize allocSize = std::min<vk::DeviceSize>(
				std::max<vk::DeviceSize>((vk::DeviceSize)pixelBufferSize, requiredSize),
				context->GetMaxMemoryAllocationSize());
#else
		vk::DeviceSize allocSize = std::min<vk::DeviceSize>(pixelBufferSize, context->GetMaxMemoryAllocationSize());
#endif
		vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer;
		if (allocSize > maxStorageBufferRange) {
			if (context->SupportsBufferDeviceAddress()) {
				NOTICE_LOG(RENDERER, "PixelBuffer allocSize %" PRIu64 " > maxStorageBufferRange %lu; will use BDA", allocSize, (unsigned long)maxStorageBufferRange);
				usage |= vk::BufferUsageFlagBits::eShaderDeviceAddressKHR;
			} else {
				NOTICE_LOG(RENDERER, "PixelBuffer allocSize %" PRIu64 " > maxStorageBufferRange %lu; capping allocSize, as the GPU doesn't support BDA", allocSize, (unsigned long)maxStorageBufferRange);
				allocSize = maxStorageBufferRange;
			}
		}
		pixelBuffer = std::make_unique<BufferData>(allocSize, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
		pixelBufferAddress =
			(usage & vk::BufferUsageFlagBits::eShaderDeviceAddressKHR)
			? context->GetDevice().getBufferAddress(vk::BufferDeviceAddressInfo{*pixelBuffer->buffer}) : 0;
		DEBUG_LOG(RENDERER, "PixelBuffer allocated. size %" PRIu64 " address %" PRIu64 "", allocSize, pixelBufferAddress);
	}
};
