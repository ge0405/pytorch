import itertools
import logging
from typing import List, Optional
from unittest.mock import patch

import sympy

import torch
from ...autotune_process import CUDABenchmarkRequest, TensorMeta
from ...ir import Buffer, IRNode, Layout
from ...utils import IndentedBuffer, unique
from ...virtualized import V
from ..common import KernelTemplate
from .cuda_kernel import CUDATemplateBuffer, CUDATemplateCaller, CUDATemplateKernel

log = logging.getLogger(__name__)


class CUDATemplate(KernelTemplate):
    index_counter = itertools.count()

    def __init__(
        self,
        name: str,
        input_nodes: List[IRNode],
        layout: Layout,
        input_reorder: Optional[List[int]] = None,
    ):
        super().__init__(name)
        self.input_nodes = input_nodes
        self.output_node = Buffer("buf_out", layout)
        self.input_reorder = input_reorder
        self.layout = layout

    def generate(
        self,
        op: "cutlass_gemm_op.GemmOperation",
        epilogue_nodes: Optional[List[IRNode]] = None,
        **kwargs,
    ) -> CUDATemplateCaller:
        kernel_name = f"cuda_{self.name}"
        with patch.object(
            V.graph, "get_dtype", self._fake_get_dtype(self.output_node)
        ), CUDATemplateKernel(
            kernel_name=kernel_name,
        ) as kernel:
            code = self.render(
                kernel=kernel, op=op, epilogue_nodes=epilogue_nodes, **kwargs
            )
            _, call_args, _ = kernel.args.python_argdefs()
            log.debug("Generated Code:\n%s", code)
            log.debug(
                "Args: cpp_argdefs: %s, python_argdefs: %s",
                kernel.args.cpp_argdefs(),
                kernel.args.python_argdefs(),
            )

        input_reorder = (
            self.input_reorder
            if self.input_reorder is not None
            else list(range(len(self.input_nodes)))
        )
        expected_args = list(
            unique(self.input_nodes[idx].get_name() for idx in input_reorder)
        )
        expected_args.extend([self.output_node.get_name()])
        assert list(call_args)[: len(expected_args)] == expected_args, (
            call_args,
            expected_args,
        )
        extra_args = V.graph.sizevars.size_hints(
            map(sympy.expand, call_args[len(expected_args) :])
        )

        kernel_hash_name = f"cuda_{self.name}_{next(self.index_counter)}"

        # create the BenchmarkRequest
        bmreq = CUDABenchmarkRequest(
            kernel_name=kernel_name,
            input_tensor_meta=TensorMeta.from_irnodes(self.input_nodes),
            output_tensor_meta=TensorMeta.from_irnodes(self.output_node),
            extra_args=extra_args,
            source_code=code,
        )

        cuda_template_buffer = CUDATemplateBuffer(
            template=self,
            op=op,
            epilogue_nodes=epilogue_nodes,
            get_workspace_size_callback=lambda: bmreq.workspace_size,
        )

        return CUDATemplateCaller(
            kernel_hash_name,
            self.name,
            bmreq,
            cuda_template_buffer,
        )

    def header(self) -> IndentedBuffer:
        res = IndentedBuffer()
        res.splice(
            """
                #include <exception>
                #include <iostream>
                #include <memory>
                #include <random>
                #include <vector>
            """
        )
        return res

    def globals(self) -> IndentedBuffer:
        res = IndentedBuffer()
        res.splice(
            """
                // We compile all models with -fvisibility=hidden. Any symbols that need to be
                // exposed in the final shared library must be declared with PT_EXPORT to make
                // them visible.
                #ifdef __GNUC__ // Applies to any compiler with GNU extensions (clang and g++)
                #define PT_EXPORT __attribute__((__visibility__("default")))
                #else
                #ifdef _WIN32
                #define PT_EXPORT __declspec(dllexport)
                #else
                #define PT_EXPORT
                #endif
                #endif
                using bfloat16 = nv_bfloat16;
            """
        )
        return res

    def render(self, **kwargs) -> str:
        raise NotImplementedError

    def can_fuse_node(self, node: IRNode) -> bool:
        """Returns true if a given node may be fused into this template.
        to be overridden
        """
        return False

    def fuse_node(self, node: IRNode):
        raise NotImplementedError()


class CUTLASSTemplate(CUDATemplate):
    def header(self) -> IndentedBuffer:
        res = super().header()
        res.splice(
            """ 
                #include "cute/tensor.hpp"
                #include "cutlass/cutlass.h"
                #include "cutlass/numeric_types.h"
                #include "cutlass/util/device_memory.h"
                #include "cutlass/tensor_ref.h"
                #include "cutlass/epilogue/collective/default_epilogue.hpp"
                #include "cutlass/epilogue/thread/linear_combination.h"
                #include "cutlass/gemm/dispatch_policy.hpp"
                #include "cutlass/gemm/collective/collective_builder.hpp"
                #include "cutlass/epilogue/collective/collective_builder.hpp"
                #include "cutlass/gemm/device/gemm_universal_adapter.h"
                #include "cutlass/gemm/kernel/gemm_universal.hpp"
                #include "cutlass/gemm/kernel/tile_scheduler.hpp"
                #include "cutlass/util/distribution.h"
                #include "cutlass/util/packed_stride.hpp"
                #include "cutlass/util/tensor_view_io.h"                
            """
        )
        return res

    def globals(self) -> IndentedBuffer:
        res = super().globals()
        res.splice(
            """
                using namespace cute;                
                #define CUTLASS_CHECK(status)                                                      \\
                {                                                                                  \\
                  cutlass::Status error = status;                                                  \\
                  if (error != cutlass::Status::kSuccess) {                                        \\
                    auto msg = std::string("[") + __FILE__ + "] Got cutlass error: " +             \\
                        cutlassGetStatusString(error) + " at: " + std::to_string(__LINE__);        \\
                    throw std::runtime_error(msg);                                                 \\
                  }                                                                                \\
                }
            """
        )
        return res

    def cute_int(self, int_str: str, var_name: str) -> str:
        res = ""
        if int_str in {"1", "1L"}:
            res = "cute::Int<1>{}"
        else:
            res = int_str

        return f"{res} /* {var_name} */"

    _DTYPE_TO_CUTLASS = {
        torch.float32: "float",
        torch.float64: "double",
        torch.float16: "cutlass::half_t",
        torch.int32: "int",
        torch.int8: "int8_t",
        torch.uint8: "uint8_t",
        torch.bool: "bool",
        torch.bfloat16: "cutlass::bfloat16_t",
    }

    def cutlass_type_cast(self, node: IRNode, ptr: str) -> str:
        if node is None:
            return ptr
        else:
            return f"({self._DTYPE_TO_CUTLASS.get(node.get_dtype())}*)({ptr})"
