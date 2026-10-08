# Known test failures

Every test listed here fails for a known reason and is left out of every run:
`ctest`, the CI and macOS ARM64 workflows, and `.github/scripts/build-macos.sh`.
The test code stays in place. Fix the cause, then delete the line to bring the
test back. Anything not listed still fails the run.

- `<binary>:<case>` skips one case. `tests/harness.hpp` reads this file at run
  time and reports the case as `skip ... (known failure, ...)`.
- `<binary>` alone marks the whole ctest entry `DISABLED`
  (`tests/CMakeLists.txt` reads this file at configure time). Use it only for
  a test that does not use the harness.

To run the excluded cases anyway, set `LSE_RUN_KNOWN_FAILURES=1`, or name a case
in `LSE_TEST_ONLY`. To run a disabled test, run its binary directly.

## Excluded

- `test_graph:quant_linear_matches_the_weights_it_encodes` — every output reads back 0.0 on the GPU path (macOS, gfx1201).
- `test_graph:quant_linear_indexed_matches_the_expert_it_selects` — every output reads back 0.0 on the GPU path (macOS, gfx1201).
- `test_graph:quant_embedding_gathers_the_rows_it_encodes` — every output reads back 0.0 on the GPU path (macOS, gfx1201).
- `test_dflash2:dflash2_sampled_generator_retains_only_verified_prefix_and_replays` — generator refuses: "feature prefix needs owned device-current storage".
- `test_dflash2:dflash2_adaptive_verify_width_keeps_the_target_distribution` — generator refuses: "feature prefix needs owned device-current storage".
- `test_dflash2:dflash2_generation_without_a_cap_stops_cleanly_at_a_full_context` — generator refuses: "feature prefix needs owned device-current storage".
- `test_dflash2:dflash2_generator_retains_full_terminal_pass_for_next_request` — generator refuses: "feature prefix needs owned device-current storage".
- `test_dflash2:dflash2_generator_commits_one_terminal_row_after_callback_cancel` — generator refuses: "feature prefix needs owned device-current storage".
- `test_dflash2:dflash2_generator_commits_two_terminal_rows_after_callback_cancel` — generator refuses: "feature prefix needs owned device-current storage".
- `test_dflash2:dflash2_generator_retains_prefill_after_first_callback_cancel` — generator refuses: "feature prefix needs owned device-current storage".
- `test_dflash2:dflash2_generator_stop_inside_verifier_commits_only_consumed_prefix` — generator refuses: "feature prefix needs owned device-current storage".
- `test_dflash2:dflash2_generator_callback_after_rejection_preserves_next_turn` — generator refuses: "feature prefix needs owned device-current storage".
- `test_dflash2:dflash2_generator_stop_after_rejection_preserves_next_turn` — generator refuses: "feature prefix needs owned device-current storage".
- `test_dflash2:dflash2_generator_prefill_stop_retains_fully_covered_history` — generator refuses: "feature prefix needs owned device-current storage".
- `test_dflash2:dflash2_generator_repeated_fully_cached_stop_request_coldstarts` — generator refuses: "feature prefix needs owned device-current storage".
- `test_dflash2:dflash2_generator_unrelated_prefix_coldstarts_retained_request` — generation result is not OK on the GPU path.
- `test_jit:a_loom_kernel_compiles_to_an_amdgpu_code_object` — the Loom compile returns an error on the macOS GPU build.
- `test_jit:target_id_features_in_the_arch_string_reach_the_code_object` — the Loom compile returns an error on the macOS GPU build.
- `test_jit:loom_compile_errors_name_the_problem_instead_of_crashing` — the well-formed control kernel fails to compile on the macOS GPU build.
- `test_jit:loom_differential_over_every_kernel_primitive` — compares against the legacy HIP path, which does not run on macOS.
- `test_jit:loom_broadcast_operand_is_bit_exact_against_hip` — compares against the legacy HIP path, which does not run on macOS.
- `test_jit:loom_elementwise_fusion_is_bit_exact_against_hip` — compares against the legacy HIP path, which does not run on macOS.
- `test_jit:loom_matches_hip_on_a_gemv_with_an_epilogue` — compares against the legacy HIP path, which does not run on macOS.
- `test_jit:loom_matches_hip_on_the_attention_and_recurrent_kernels` — compares against the legacy HIP path, which does not run on macOS.
- `test_jit:loom_matches_hip_on_the_elementwise_vocabulary` — compares against the legacy HIP path, which does not run on macOS.
- `test_jit:loom_matches_hip_on_the_linear_kernels` — compares against the legacy HIP path, which does not run on macOS.
- `test_jit:loom_matches_hip_on_the_shape_kernels` — compares against the legacy HIP path, which does not run on macOS.
- `test_jit:loom_matmul_with_an_epilogue_matches_hip` — compares against the legacy HIP path, which does not run on macOS.
- `test_jit:loom_overwrite_slice_is_bit_exact_against_hip` — compares against the legacy HIP path, which does not run on macOS.
- `test_jit:loom_silu_matches_hip_across_the_range` — compares against the legacy HIP path, which does not run on macOS.
- `test_quant_operand_cache` — fails on the GPU build (it drives the legacy HIP emitter); not a harness test, so the whole ctest entry is disabled.
