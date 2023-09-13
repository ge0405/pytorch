from typing import Callable, Dict, List, Optional

from ...autotune_process import CUDABenchmarkRequest
from ...ir import IRNode, TemplateBuffer, TensorBox
from ...select_algorithm import ChoiceCaller
from ...utils import sympy_product
from ...virtualized import V

from ..common import IndentedBuffer, Kernel, OpOverrides
from ..cpp import CppPrinter, DTYPE_TO_CPP

cexpr = CppPrinter().doprint


def _normalize_idx(index: int, total_length: int) -> int:
    return index if index >= 0 else index + total_length


class CUDAKernel(Kernel):
    """
    Kernels defined by C++ CUDA.
    """

    overrides = OpOverrides  # type: ignore[assignment]


class CUDATemplateKernel(CUDAKernel):
    """
    Template kernels defined by C++ CUDA.
    """

    _EXTRA_CPP_ARGS = "size_t* workspace_size, uint8_t* workspace, cudaStream_t stream"

    def __init__(self, kernel_name):
        super().__init__()
        self.kernel_name = kernel_name
        # Mapping from arg name to IRNode.
        self.named_nodes: Dict[str, IRNode] = {}

    def arg_name(self, node: IRNode) -> Optional[str]:
        """
        Returns arg name of a given input or output node.
        """

        if node is None:
            return None
        return {**self.args.input_buffers, **self.args.output_buffers}.get(
            node.get_name(), None
        )

    def check_not_null(self, node: IRNode) -> str:
        """
        Generates code to check that a node is not null.
        """

        if node is None:
            return ""

        size_str = self.size(node, 0, -1)
        name_str = self.arg_name(node)
        if name_str is None:
            return ""

        res = IndentedBuffer(initial_indent=2)
        res.tabwidth = 1
        res.splice(
            f"""
            {{
              if (!{name_str}) {{
                int64_t {name_str}_size = {size_str};
                if ({name_str}_size > 0) {{
                  throw std::runtime_error("input {name_str} is null but size is not 0!");
                }}
              }}
            }}
            """
        )
        return res.getvalue()

    def def_kernel(
        self,
        inputs: List[IRNode],
        outputs: List[IRNode],
        names_str: str = "",
        input_reorder: Optional[List[int]] = None,
    ) -> str:
        """
        Hook called from template code to generate function def and
        needed args.

        inputs / outputs: List of input / output IRNodes. Note that IRNode can be None for optional arguments.
        names_str: Comma separated list of input + output argument names.
        input_reorder: The actual order of input nodes.
                       e.g. The template might have input argument defined as [X, W, Bias],
                       and the actual input passed into this template could be [Bias, X, W].
                       In this case, the `input_reorder` would be [2, 0, 1].
        """

        names = [x.strip() for x in names_str.strip().split(",")]
        if len(inputs) + len(outputs) != len(names):
            raise RuntimeError(
                f"{len(inputs) + len(outputs)=} != {len(names)=}, {inputs=}, {outputs=}, {names=}"
            )

        if input_reorder is not None:
            assert len(inputs) == len(input_reorder)
        else:
            input_reorder = list(range(len(inputs)))

        for idx in input_reorder:
            name = names[idx]
            node = inputs[idx]
            if node is not None:
                self.named_nodes[name] = node
                self.args.input_buffers[node.get_name()] = name

        for name, node in zip(names[len(inputs) : len(inputs) + len(outputs)], outputs):
            if node is not None:
                self.named_nodes[name] = node
                self.args.output_buffers[node.get_name()] = name

        arg_defs, *_ = self.args.cpp_argdefs()
        return f"PT_EXPORT int {self.kernel_name}({', '.join(arg_defs)}, {self._EXTRA_CPP_ARGS})"

    def call_kernel(self, name: str, node: "CUDATemplateBuffer") -> None:
        """
        Generates code to call the kernel through V.graph.wrapper_code.

        name: Name of kernel function.
        node: The IRNode which represents the kernel.
        """

        wrapper = V.graph.wrapper_code
        _, call_args, _ = self.args.python_argdefs()
        # dynamo wraps unspec variable as 0d CPU tensor, need convert to scalar
        for i in range(len(call_args)):
            if V.graph.is_unspec_arg(call_args[i]):
                call_args[i] = call_args[i] + ".item()"
            else:
                call_args[i] = f"c_void_p({call_args[i]}.data_ptr())"

        # workspace_size ptr is NULL to mark this call is not intended for retrieving workspace_size.
        # workspace_size should have already been retrieved prior to this call.
        call_args.append("None")

        if node.get_workspace_size() > 0:
            call_args.append(f"c_void_p({node.get_name()}_workspace.data_ptr())")
        else:
            call_args.append("None")

        wrapper.generate_kernel_call(
            name,
            call_args,
            device_index=V.graph.scheduler.current_device.index,
            cuda=True,
            triton=False,
        )

    def dtype(self, node: IRNode) -> str:
        """
        Generates code which represents dtype of a given node.
        """

        if node is None:
            return "void"
        return DTYPE_TO_CPP.get(node.get_layout().dtype)

    def offset(self, node: IRNode) -> str:
        """
        Generates code which represents offset of a given node.
        """

        if node is None:
            return "0"
        return str(node.get_layout().offset)

    def ptr(self, node: IRNode) -> str:
        """
        Generates code which represents pointer of a given node.
        """

        if node is None:
            return "nullptr"
        arg_name = self.arg_name(node)
        if arg_name is None:
            return "nullptr"
        offset = self.offset(node)
        return arg_name if offset == "0" else f"{arg_name} + {offset}"

    def size(
        self,
        node: IRNode,
        start_index: int,
        end_index: Optional[int] = None,
        default_value: int = 0,
    ) -> str:
        """
        Hook called from template code to get the size of an arg.
        Generates code which represents size of a given node in [start_index, end_index).
        If node is None, returns default_value.

        TODO: Will add needed args to pass it in if it is dynamic.
        """

        if node is None:
            return str(default_value)

        start_index = _normalize_idx(start_index, len(node.get_size()))
        if end_index is None:
            end_index = start_index
        end_index = _normalize_idx(end_index, len(node.get_size()))

        sizes = node.get_size()[start_index : end_index + 1]
        if len(sizes) == 0:
            return str(default_value)

        val = sympy_product(sizes)
        return cexpr(self.rename_indexing(val))

    def stride(self, node: IRNode, index: int, default_value: int = 0) -> str:
        """
        Hook called from template code to get the stride of an arg.
        Generates code which represents stride of a given node at index.
        If node is None, returns default_value.

        TODO: Will add needed args to pass it in if it is dynamic.
        """

        if node is None:
            return str(default_value)

        index = _normalize_idx(index, len(node.get_size()))
        if index < 0:
            return str(default_value)

        stride = node.get_stride()[index]
        return cexpr(self.rename_indexing(stride))

    def row_or_column_stride(self, node: IRNode, default_value: int = 0) -> str:
        """
        Hook called from template code to get the row or column stride of an arg.
        This is required by some CUTLASS 2.X APIs.
        If the node is in row_major, it returns stride[-2].
        If the node is in column_major, it returns stride[-1].

        TODO: Will add needed args to pass it in if it is dynamic.
        """

        if node is None or len(node.get_stride()) < 2:
            return str(default_value)

        stride0 = node.get_stride()[-1]
        stride1 = node.get_stride()[-2]
        if stride0 == 1:
            return cexpr(self.rename_indexing(stride1))
        elif stride1 == 1:
            return cexpr(self.rename_indexing(stride0))
        else:
            raise RuntimeError(
                f"At least 1 stride should be 1. Strides: {node.get_stride()=}"
            )


class CUDATemplateBuffer(TemplateBuffer):
    def __init__(
        self,
        template: "CUDATemplate",
        op: "cutlass_gemm_op.GemmOperation",  # TODO: Update type once we have other op-types
        epilogue_nodes: Optional[
            List[IRNode]
        ] = None,  # We need a new instance of this op every time we fuse
        get_workspace_size_callback: Callable[
            [], int
        ] = None,  # This is not known at construction time, so we need a callback
        **render_kwargs,  # passed through to template_node.render
    ):
        if epilogue_nodes is None:
            epilogue_nodes = []
        input_nodes = self._merge_inputs(epilogue_nodes, template)
        super().__init__(template.layout, input_nodes, self._make_kernel_render)
        self.template = template
        self.op = op
        self._epilogue_nodes = epilogue_nodes
        # TODO: Once we support non-pointwise fusions, layout might be modified by epilogues
        self.layout = template.layout
        self._render_kwargs = render_kwargs
        # Global memory (in bytes) needed for this template.
        self._get_workspace_size = get_workspace_size_callback

    @staticmethod
    def _merge_inputs(epilogue_nodes, template):
        # Merge all inputs, including extra inputs from epilogue_nodes
        # input nodes are not hashable, so we cannot directly place them in sets
        template_input_id_set = set([id(node) for node in template.input_nodes])
        total_input_id_set = set(template_input_id_set)
        extra_inputs = []
        for epilogue_node in epilogue_nodes:
            for node in epilogue_node.input_nodes:
                if id(node) not in total_input_id_set:
                    extra_inputs.append(node)
                    total_input_id_set.add(id(node))
        input_nodes = list(template.input_nodes) + extra_inputs
        return input_nodes

    @property
    def epilogue_nodes(self):
        # read-only property to signal it should not be mutated
        return self._epilogue_nodes

    @property
    def workspace_size(self):
        return self.get_workspace_size()

    def get_workspace_size(self):
        return (
            self._get_workspace_size() if not (self._get_workspace_size is None) else 0
        )

    def _kernel_render(self):
        kernel = CUDATemplateKernel(
            kernel_name="KERNEL_NAME",
        )
        return self.template.render(
            kernel=kernel, output_node=self, op=self.op, **self._render_kwargs
        )

    def _make_kernel_render(self, output_node):
        assert output_node is self
        assert self.workspace_size >= 0
        return self._kernel_render

    def get_scheduler_node_class(self):
        from torch._inductor.scheduler import CUDASchedulerNode

        return CUDASchedulerNode

    def can_fuse_epilogue(self, node):
        self.template.can_fuse_epilogue(node)


class CUDATemplateCaller(ChoiceCaller):
    """
    CUDATemplateCaller

    This class represents a caller for CUDA template kernels. It is a subclass of ChoiceCaller.
    Attributes:
        name (str): The name of the caller.
        category (str): The category of the caller.
        bmreq (CUDABenchmarkRequest): The benchmark request for the caller.
        template_buffer (CUDATemplateBuffer): The template buffer for the caller.
    """

    def __init__(
        self,
        name: str,
        category: str,
        bmreq: CUDABenchmarkRequest,
        template_buffer: CUDATemplateBuffer,
    ):
        super().__init__(
            name, template_buffer.template.input_nodes, template_buffer.layout
        )
        self.category = category
        self.bmreq = bmreq

        self.template_buffer = template_buffer

    def benchmark(self, *args, out) -> float:
        assert self.bmreq is not None
        return self.bmreq.benchmark(*args, output_tensor=out)

    def __str__(self):
        return f"CUDATemplateCaller(source_file={self.bmreq.source_file})"

    def call_name(self) -> str:
        return f"cuda_template_kernels.{self.name}"

    def hash_key(self) -> str:
        return "-".join(
            [
                self.category,
                self.bmreq.hash_key,
            ]
        )

    def output_node(self) -> TensorBox:
        return TensorBox.create(self.template_buffer)
