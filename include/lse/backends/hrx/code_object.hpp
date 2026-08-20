// The MEASUREMENT and FACTS halves of this backend, for AMD code objects.
//
// Not under hipc/ or loomc/: an AMDGPU code object is an AMDGPU code object
// whichever generator wrote it, and both of this backend's compilers hand
// their bytes to the same reader here. The note both emit is the ELF
// `amdhsa.kernels` map, read through amd_comgr rather than by walking the ELF
// ourselves, so an ELF or metadata-version change is the toolchain's problem.
#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "lse/backend/backend.hpp"
#include "lse/backend/census.hpp"
#include "lse/backend/resources.hpp"
#include "lse/core/status.hpp"

namespace lse::backend {

// What the object says about every kernel it defines. An object whose note is
// missing or unreadable yields an empty vector and an ok status: "this
// toolchain reports nothing" is an answer, and failing a compile over it would
// trade a working kernel for a diagnostic.
//
// Not every key is present on every producer. Measured: loomc's code objects
// carry vgpr/sgpr/segment sizes and their own required workgroup size but no
// spill counts at all, while comgr's carry the spill counts even when zero.
// Absent keys come back kUnknown, never 0.
[[nodiscard]] std::vector<KernelResources> read_code_object_resources(
    std::span<const std::byte> object);

// The target identification string the object states it was built for
// ("amdgcn-amd-amdhsa--gfx1151"), or empty when it does not say. Read from the
// object rather than passed in, so a reader cannot be handed the wrong one.
[[nodiscard]] std::string read_code_object_target(
    std::span<const std::byte> object);

// What the object's instructions add up to, one entry per kernel it defines,
// by disassembling it with the toolchain's own disassembler.
//
// This is the MEASUREMENT half applied to the emitted body rather than to the
// register allocator's report: it answers "what did the compiler actually
// write", which is the only source that can contradict what the emitter meant
// to write. Empty when this build has no disassembler or the object carries no
// code — an answer, not a failure.
[[nodiscard]] std::vector<KernelCensus> read_code_object_census(
    std::span<const std::byte> object);

// What the compiler's own target table says about `arch` ("gfx1151", or a
// target id with feature suffixes). Everything unknown when this build has no
// comgr, and individually unknown for any key the table omits.
//
// Two keys the table does carry are deliberately NOT read: MaxWavesPerCU and
// EUsPerCU are byte-identical across every AMD target from gfx1030 to gfx1201
// to gfx942, so they are compiler-model constants rather than per-target
// facts, and the 40 waves/CU they claim is contradicted by the 32 the device's
// own runtime reports. Leaving them out is the point; do not "fix" the
// omission.
[[nodiscard]] ArchFacts query_isa_facts(std::string_view arch);

// Capacity for the device `info` describes, best source first: the compiler's
// own target table, then what the device runtime itself already answered, then
// this project's per-family rows, then nothing. Each field carries which of
// those answered it, so a declared number is never mistaken for a queried one
// and an unanswered one is never a zero.
[[nodiscard]] ArchFacts arch_facts_for(const DeviceInfo& info);

}  // namespace lse::backend
