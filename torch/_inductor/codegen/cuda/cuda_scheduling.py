from unittest.mock import patch

from ... import config
from ...codecache import code_hash, get_path
from ...ir import CUDATemplateBuffer
from ...utils import get_fused_kernel_name, get_kernel_metadata
from ...virtualized import V

from ..common import IndentedBuffer
from ..triton import TritonScheduling

from . import cutlass_epilogue_gen
from .cuda_template import CUDATemplate
from .cutlass_epilogue_gen import CutlassEpilogueFormatterHandler


class CUDAScheduling(TritonScheduling):
    """
    Final codegen for CUDAKernels.
    """

    def define_kernel(self, src_code: str, node_schedule) -> str:
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            fused_name = (
                get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
                if config.triton.descriptive_names
                else ""
            )
            kernel_name = "_".join(["cuda", fused_name, wrapper.next_kernel_suffix()])
            # use the original src_code as the key
            wrapper.src_to_kernel[src_code] = kernel_name
            src_code = src_code.replace("KERNEL_NAME", kernel_name)

            _, _, kernel_path = get_path(code_hash(src_code), "py")

            compile_wrapper = IndentedBuffer()
            compile_wrapper.writeline("async_compile.cuda(r'''")
            compile_wrapper.splice(src_code, strip=True)
            compile_wrapper.writeline("''', 'so')")

            metadata_comment = f"# kernel path: {kernel_path}"
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment += "\n" + origins + "\n" + detailed_origins
            wrapper.define_kernel(
                kernel_name, compile_wrapper.getvalue(), metadata_comment
            )
        return kernel_name

    def codegen_template(self, template_node, epilogue_nodes):
        """
        Codegen a CUDA template
        """
        _, (numel, rnumel) = template_node.group
        assert rnumel == 1
        assert isinstance(template_node.node, CUDATemplateBuffer)
        kernel, render = template_node.node.make_kernel_render(template_node.node)
        epilogue_init_list = []
        epilogue_param_list = []
        with kernel:
            for node in [template_node, *epilogue_nodes]:
                node.mark_run()
            src_code = render()

        with V.set_kernel_handler(kernel):
            node_schedule = [template_node, *epilogue_nodes]
            kernel_name = self.define_kernel(src_code, node_schedule)
        self.codegen_comment(node_schedule)
        kernel.call_kernel(kernel_name, template_node.node)
        V.graph.removed_buffers |= kernel.removed_buffers
        self.scheduler.free_buffers()
