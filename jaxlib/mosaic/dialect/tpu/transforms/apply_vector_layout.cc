#include <array>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

// TODO(tlongeri): shorter includes for mlir/llvm (e.g. like layout.h)
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "jaxlib/mosaic/dialect/tpu/layout.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/array.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

// TODO(tlongeri): Prefer returning failure over CHECKs. In particular, be more
// consistent about this for layout null checks in rules.

// TODO: Instead of CHECK_EQs, can we do something like TF_RET_CHECK but with
// MLIR diagnostics?
// e.g.
// #define MLIR_RET_CHECK_EQ(a, b, diagnostic) \
//   do { \
//     const auto a_ = a; \
//     const auto b_ = b; \
//     if (LLVM_UNLIKELY(a_ != b_)) { \
//       return diagnostic << "Check failed: " << #a << " != " << #b << "(" <<
//       a_  << " vs. " << b_ << ")"; \
//     } \
//   } while (false)

#define FAILUREOR_ASSIGN_OR_RETURN_IMPL(failureor, lhs, rhs) \
  auto failureor = rhs;                                      \
  if (failed(failureor)) {                                   \
    return failure();                                        \
  }                                                          \
  lhs = std::move(failureor).value();
#define FAILUREOR_ASSIGN_OR_RETURN(lhs, rhs) \
  FAILUREOR_ASSIGN_OR_RETURN_IMPL(           \
      TF_STATUS_MACROS_CONCAT_NAME(failureor, __COUNTER__), lhs, rhs)
#define NYI(msg)                            \
  op->emitOpError("not implemented: " msg); \
  return failure();

namespace mlir::tpu {
// TODO(tlongeri): Maybe just roll our own multi-dimensional array instead of
// using XLA's? There's too much glue for going from/to ArrayRef.

#define GEN_PASS_DECL_APPLYVECTORLAYOUTPASS
#define GEN_PASS_DEF_APPLYVECTORLAYOUTPASS
#include "jaxlib/mosaic/dialect/tpu/tpu_passes.h.inc"

struct RewriteContext {
  func::FuncOp func;
  OpBuilder &builder;
  // TODO(tlongeri): target_shape should be determined from hardware_generation
  const int hardware_generation;
  const std::array<int64_t, 2> target_shape;
};

RollVectorsOp assemble(RewriteContext &ctxt, VectorType vty,
                       const VectorLayout &layout, xla::Array<Value> vals);
FailureOr<xla::Array<Value>> disassemble(RewriteContext &ctxt,
                                         const VectorLayout &layout, Value val);
namespace {

template <bool adjust_bool = false>
FailureOr<int8_t> getTypeBitwidth(Type ty) {
  if (auto integer_ty = dyn_cast<IntegerType>(ty)) {
    const unsigned width = integer_ty.getWidth();
    if constexpr (adjust_bool) {
      // We store only one i1 per vreg element.
      return width == 1 ? 32 : width;
    } else {
      return width;
    }
  }
  if (auto f32_ty = dyn_cast<Float32Type>(ty)) {
    return 32;
  }
  if (auto bf16_ty = dyn_cast<BFloat16Type>(ty)) {
    return 16;
  }
  return emitError(UnknownLoc::get(ty.getContext()), "Unsupported type: ")
         << ty;
}

FailureOr<VectorType> getNativeVregType(
    Type elem_ty, const std::array<int64_t, 2> target_shape) {
  FAILUREOR_ASSIGN_OR_RETURN(const int8_t bitwidth,
                             getTypeBitwidth<true>(elem_ty));
  if (bitwidth == 32) {
    return VectorType::get(target_shape, elem_ty);
  }
  // bitwidth != 32
  return VectorType::get({target_shape[0], target_shape[1], 32 / bitwidth},
                         elem_ty);
}

LogicalResult elementwise_op_rule(
    RewriteContext &ctxt, Operation &op, ArrayRef<Layout> layout_in,
    const Layout &layout_out,
    std::function<FailureOr<Operation *>(RewriteContext &, ArrayRef<Value>)>
        factory) {
  CHECK_EQ(layout_in.size(), op.getNumOperands());
  CHECK_GT(layout_in.size(), 0);
  if (!(layout_out.has_value() && llvm::all_of(layout_in, [&](const Layout &l) {
          return l.has_value();
        }))) {
    return op.emitOpError("null layout in elementwise operation");
  }
  if (!llvm::all_of(layout_in, [&](const Layout &l) {
        return l->generalizes(*layout_out, std::nullopt, ctxt.target_shape);
      })) {
    return op.emitOpError("incompatible layouts in elementwise operation");
  }
  const unsigned num_operands = op.getNumOperands();
  CHECK_EQ(layout_in.size(), num_operands);
  SmallVector<xla::Array<Value>> in_tile_arrays;
  in_tile_arrays.reserve(num_operands);
  for (unsigned i = 0; i < num_operands; ++i) {
    FAILUREOR_ASSIGN_OR_RETURN(
        xla::Array<Value> tile_array,
        disassemble(ctxt, *layout_in[i], op.getOperand(i)));
    in_tile_arrays.emplace_back(std::move(tile_array));
  }

  // Note that we have to broadcast to handle replicate dimensions.
  SmallVector<int64_t> broadcasted_shape(
      ArrayRef(in_tile_arrays[0].dimensions().data(),
               in_tile_arrays[0].dimensions().size()));
  for (size_t i = 1; i < num_operands; ++i) {
    ArrayRef shape(in_tile_arrays[i].dimensions().data(),
                   in_tile_arrays[i].dimensions().size());
    CHECK(OpTrait::util::getBroadcastedShape(broadcasted_shape, shape,
                                             broadcasted_shape));
  }

  // TODO(tlongeri): Can we avoid initializing the array before filling values?
  xla::Array<Value> out_tile_array(broadcasted_shape);
  absl::Status status =
      out_tile_array.EachStatus([&](absl::Span<const int64_t> idx, Value *v) {
        SmallVector<Value> operands(num_operands);
        for (unsigned i = 0; i < num_operands; ++i) {
          // Handle indices for broadcasted dimensions
          SmallVector<int64_t> operand_idx(toArrayRef(idx));
          for (unsigned j = 0; j < idx.size(); ++j) {
            if (in_tile_arrays[i].dimensions()[j] == 1) {
              operand_idx[j] = 0;
            }
          }
          operands[i] = in_tile_arrays[i](operand_idx);
        }
        FailureOr<Operation *> failure_or_tile_op = factory(ctxt, operands);
        if (failed(failure_or_tile_op)) {
          return absl::InvalidArgumentError("");
        }
        Operation *tile_op = *failure_or_tile_op;
        CHECK(tile_op);
        CHECK_EQ(tile_op->getNumResults(), 1);
        *v = tile_op->getResult(0);
        return absl::OkStatus();
      });
  if (!status.ok()) {
    return failure();
  }
  auto ty = cast<VectorType>(op.getResult(0).getType());
  op.replaceAllUsesWith(
      assemble(ctxt, ty, *layout_out, std::move(out_tile_array)));
  op.erase();
  return success();
}

// Helper for index_sequence expansion
template <typename T, std::size_t>
using Wrapper = T;

template <std::size_t... I>
LogicalResult elementwise_op_rule_unpacked_impl(
    RewriteContext &ctxt, Operation &op, const ArrayRef<Layout> layout_in,
    const Layout &layout_out,
    std::function<FailureOr<Operation *>(RewriteContext &ctxt,
                                         Wrapper<Value, I>...)>
        factory,
    std::index_sequence<I...>) {
  return elementwise_op_rule(
      ctxt, op, layout_in, layout_out,
      [&](RewriteContext &ctxt,
          ArrayRef<Value> operands) -> FailureOr<Operation *> {
        if (operands.size() != sizeof...(I)) {
          return failure();
        }
        return factory(ctxt, operands[I]...);
      });
}

template <std::size_t NumOperands, typename Func>
LogicalResult elementwise_op_rule_unpacked(RewriteContext &ctxt, Operation &op,
                                           const ArrayRef<Layout> layout_in,
                                           const Layout &layout_out,
                                           Func factory) {
  return elementwise_op_rule_unpacked_impl(
      ctxt, op, layout_in, layout_out, std::move(factory),
      std::make_index_sequence<NumOperands>());
}

using rule_type = std::function<LogicalResult(
    RewriteContext &, Operation &, ArrayRef<Layout>, const Layout &)>;

template <typename Op, std::size_t NumOperands>
std::pair<StringRef, rule_type> rules_elementwise_op_entry() {
  return {
      Op::getOperationName(),
      [](RewriteContext &ctxt, Operation &op, const ArrayRef<Layout> layout_in,
         const Layout &layout_out) -> LogicalResult {
        return elementwise_op_rule_unpacked<NumOperands>(
            ctxt, op, layout_in, layout_out,
            [&](RewriteContext &ctxt,
                auto... operands) -> FailureOr<Operation *> {
              return ctxt.builder.create<Op>(op.getLoc(), operands...)
                  .getOperation();
            });
      }};
}

const llvm::StringMap<rule_type> &rules() {
  static auto rules = new llvm::StringMap<rule_type>{
      rules_elementwise_op_entry<arith::AddFOp, 2>(),
      rules_elementwise_op_entry<arith::AddIOp, 2>(),
      rules_elementwise_op_entry<arith::SubFOp, 2>(),
      rules_elementwise_op_entry<arith::SubIOp, 2>(),
      rules_elementwise_op_entry<arith::MulFOp, 2>(),
      rules_elementwise_op_entry<arith::MulIOp, 2>(),
      rules_elementwise_op_entry<arith::DivFOp, 2>(),
      rules_elementwise_op_entry<arith::DivSIOp, 2>(),
      rules_elementwise_op_entry<arith::RemSIOp, 2>(),
      rules_elementwise_op_entry<arith::MaxFOp, 2>(),
      rules_elementwise_op_entry<arith::MinFOp, 2>(),
      rules_elementwise_op_entry<arith::SelectOp, 3>(),
      // TODO(tlongeri) arith::IndexCastOp
      rules_elementwise_op_entry<arith::AndIOp, 2>(),
      rules_elementwise_op_entry<arith::OrIOp, 2>(),
      rules_elementwise_op_entry<arith::NegFOp, 1>(),
      rules_elementwise_op_entry<arith::XOrIOp, 2>(),
      rules_elementwise_op_entry<arith::ShLIOp, 2>(),
      rules_elementwise_op_entry<arith::ShRUIOp, 2>(),
      rules_elementwise_op_entry<math::ExpOp, 1>(),
      rules_elementwise_op_entry<math::CosOp, 1>(),
      rules_elementwise_op_entry<math::SinOp, 1>(),
      rules_elementwise_op_entry<math::PowFOp, 1>(),
      rules_elementwise_op_entry<math::RsqrtOp, 1>(),
      rules_elementwise_op_entry<math::TanhOp, 1>(),
  };
  return *rules;
}
}  // namespace

// Get the layout from a VectorLayoutAttr or StringAttr.
// Returns kNoLayout on null attribute and failure on invalid attribute.
mlir::FailureOr<Layout> getLayoutFromAttr(Attribute attr) {
  if (attr == nullptr) {
    return kNoLayout;
  }

  if (auto layout_attr = dyn_cast<VectorLayoutAttr>(attr)) {
    return layout_attr.getLayout();
  }

  // TODO(tlongeri): StringAttr support was only added temporarily to avoid
  // having Python bindings for VectorLayoutAttr. Remove this once we get rid
  // of the Python implementation
  if (auto string_attr = dyn_cast<StringAttr>(attr)) {
    StringRef str = string_attr.getValue();
    if (!str.consume_front("#tpu.vpad<\"")) {
      return failure();
    }
    if (str.consume_front("none")) {
      return kNoLayout;
    }
    if (auto layout = VectorLayout::parse(&str)) {
      return layout;
    }
    return failure();
  }

  return failure();
}

// Returns empty vector on null attribute
FailureOr<SmallVector<Layout>> getLayoutArrayFromAttr(const Attribute attr) {
  if (const auto array_attr = dyn_cast_if_present<ArrayAttr>(attr)) {
    SmallVector<Layout> out_layouts;
    out_layouts.reserve(array_attr.size());
    for (const Attribute a : array_attr) {
      FAILUREOR_ASSIGN_OR_RETURN(const Layout layout, getLayoutFromAttr(a));
      out_layouts.push_back(layout);
    }
    return out_layouts;
  }
  return SmallVector<Layout>{};
}

// TODO(tlongeri): Unify with infer_vector_layout.cc's getOutLayout.
FailureOr<SmallVector<Layout>> getOutLayout(Operation &op) {
  // TODO(tlongeri): non-array attribute path should be removed after tests are
  // updated
  FailureOr<Layout> failure_or_layout =
      getLayoutFromAttr(op.getAttr("out_layout"));
  if (succeeded(failure_or_layout)) {
    return SmallVector<Layout>{failure_or_layout.value()};
  }
  FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> out_layout,
                             getLayoutArrayFromAttr(op.getAttr("out_layout")));
  if (out_layout.size() != op.getNumResults()) {
    return failure();
  }
  return out_layout;
}

FailureOr<SmallVector<Layout>> getInLayout(Operation &op) {
  FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> in_layout,
                             getLayoutArrayFromAttr(op.getAttr("in_layout")));
  if (in_layout.size() != op.getNumOperands()) {
    return failure();
  }
  return in_layout;
}

template <typename T>
ArrayRef<T> XlaArrayToFlatArrayRef(xla::Array<T> xla_array) {
  return ArrayRef<T>(xla_array.data(), xla_array.num_elements());
}

template <typename T, typename Range>
xla::Array<T> XlaArrayFromShapeAndValues(ArrayRef<int64_t> sizes, Range vals) {
  // TODO(tlongeri): is there no way to avoid default initialization in the
  // constructor?
  xla::Array<T> arr(sizes);
  arr.SetValues(vals);
  return arr;
}
RollVectorsOp assemble(RewriteContext &ctxt, VectorType vty,
                       const VectorLayout &layout,
                       const xla::Array<Value> vals) {
  CHECK(vals.dimensions() ==
        layout.tileArrayShape(vty.getShape(), ctxt.target_shape));
  ValueRange values;
  CHECK_GT(vals.num_elements(), 0);
  Location loc = vals.begin()->getLoc();
  DCHECK(llvm::all_of(vals, [&](Value v) { return v.getLoc() == loc; }));
  auto op = ctxt.builder.create<RollVectorsOp>(loc, vty,
                                               XlaArrayToFlatArrayRef(vals));
  op->setAttr("out_layout", ctxt.builder.getAttr<VectorLayoutAttr>(layout));
  return op;
}

// Disassemble an MLIR vector into an ndarray of native vectors.
//
// Args:
//   layout: The layout of val. Used to determine the unrolling into
//     native-shaped vectors.
//   val: Value to disassemble. Must be of type VectorType.
//
// Returns:
//   An ndarray of MLIR values representing the tiling of val given by layout.
FailureOr<xla::Array<Value>> disassemble(RewriteContext &ctxt,
                                         const VectorLayout &layout,
                                         const Value val) {
  const auto vty = cast<VectorType>(val.getType());
  const SmallVector<int64_t> tiles_shape =
      layout.tileArrayShape(vty.getShape(), ctxt.target_shape);
  const auto op_result = dyn_cast<OpResult>(val);
  if (op_result == nullptr) {
    return failure();
  }
  Operation *const op = op_result.getOwner();
  CHECK(op != nullptr);
  const unsigned res_idx = op_result.getResultNumber();
  FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> def_layouts,
                             getOutLayout(*op));
  const Layout def_layout = def_layouts[res_idx];
  if (!def_layout.has_value()) {
    return op->emitOpError("found null out_layout") << val;
  }
  // TODO(tlongeri): They should actually be equal, which means the layout
  //                 parameter is redundant.
  CHECK(def_layout->generalizes(layout, std::nullopt, ctxt.target_shape));
  SmallVector<int64_t> def_layout_shape =
      def_layout->tileArrayShape(vty.getShape(), ctxt.target_shape);
  if (auto roll_vectors_op = dyn_cast<RollVectorsOp>(op)) {
    return XlaArrayFromShapeAndValues<Value>(def_layout_shape,
                                             roll_vectors_op->getOperands());
  }
  if (auto contraction_op = dyn_cast<vector::ContractionOp>(op)) {
    const int64_t num_vectors = ShapedType::getNumElements(tiles_shape);
    FAILUREOR_ASSIGN_OR_RETURN(
        VectorType vreg_ty,
        getNativeVregType(vty.getElementType(), ctxt.target_shape));
    // TODO(tlongeri): nicer way of doing ValueTypeRange?
    Operation *const u = ctxt.builder.create<UnrollVectorsOp>(
        val.getLoc(), SmallVector<Type>(num_vectors, vreg_ty), val);
    return XlaArrayFromShapeAndValues<Value>(def_layout_shape, u->getResults());
  }
  return op->emitOpError("unimplemented: ") << val;
}

// Changes the layout of a vector value.
//
// Arguments:
//   v: The value to relayout. Must be of type VectorType.
//   src: The current layout of v.
//   dst: The target layout of v.
//
// Returns:
//   A new MLIR vector value, laid out as requested by dst.
// TODO(apaszke): Test this function properly
FailureOr<Value> relayout(RewriteContext &ctxt, Value v, VectorLayout src,
                          const VectorLayout &dst) {
  const int8_t bitwidth = src.bitwidth();
  if (bitwidth != dst.bitwidth()) {
    return emitError(v.getLoc(), "Can't change bitwidth during a relayout");
  }
  const int packing = src.packing();
  VectorType vty = cast<VectorType>(v.getType());
  FAILUREOR_ASSIGN_OR_RETURN(xla::Array<Value> src_tiles,
                             disassemble(ctxt, src, v));
  SmallVector<int64_t> dst_tiles_shape =
      dst.tileArrayShape(vty.getShape(), ctxt.target_shape);
  if (src.generalizes(dst, vty.getShape(), ctxt.target_shape)) {
    return assemble(ctxt, vty, dst, std::move(src_tiles)).getResult();
  }
  if (!src.offsets()[0].has_value() && !src.offsets()[1].has_value()) {
    // A fully replicated value is always easy to relayout
    // TODO(apaszke): It would be nice to be able to assert this here, but
    // given replicated values our rules can introduce equivalent expressions.
    // assert all(t is src_tiles_list[0] for t in src_tiles_list)
    xla::Array<Value> dst_tiles(
        /*sizes=*/dst.tileArrayShape(vty.getShape(), ctxt.target_shape),
        /*value=*/src_tiles.data()[0]);
    return assemble(ctxt, vty, dst, std::move(dst_tiles)).getResult();
  }
  // Try to reconcile differences in implicit dim.
  if (src.implicit_dim() != dst.implicit_dim()) {
    const ArrayRef<int64_t> shape = vty.getShape();
    if (dst.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
        shape[shape.size() - xla::to_underlying(src.implicit_dim())] == 1) {
      src = VectorLayout(src.bitwidth(), src.offsets(), src.tiling(),
                         VectorLayout::ImplicitDim::kNone);
    }
  }

  // TODO(apaszke): Generalize retiling to general 16-bit types (might need to
  // use a different unpacking op).
  VectorType vreg_f32 =
      VectorType::get(ctxt.target_shape, ctxt.builder.getF32Type());
  // (8,128) -> (16,128) tiling change for packed 16-bit types.
  if (src.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      dst.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      vty.getElementType() == ctxt.builder.getBF16Type() &&
      src.offsets() == dst.offsets() &&
      src.tiling() == std::array<int64_t, 2>{8, 128} &&
      dst.tiling() == std::array<int64_t, 2>{16, 128}) {
    const VectorLayout new_src(src.bitwidth(), src.offsets(), dst.tiling());
    xla::Array<Value> src_tiles_retiled(
        new_src.tileArrayShape(vty.getShape(), ctxt.target_shape));
    src_tiles_retiled.Each([&](absl::Span<const int64_t> idx, Value *tile) {
      SmallVector<int64_t> src_idx(idx.begin(), idx.end());
      src_idx[src_idx.size() - 2] *= 2;
      src_idx[src_idx.size() - 1] /= 2;
      Value src_row1 = src_tiles(src_idx);
      if (src_idx[src_idx.size() - 2] + 1 <
          src_tiles.dim(src_tiles.num_dimensions() - 2)) {
        ++src_idx[src_idx.size() - 2];
      }
      Value src_row2 = src_tiles(src_idx);
      const int vreg_part = idx[idx.size() - 1] % 2;
      auto half_row1 = ctxt.builder.create<tpu::UnpackSubelementsOp>(
          v.getLoc(), vreg_f32, src_row1, vreg_part);
      auto half_row2 = ctxt.builder.create<tpu::UnpackSubelementsOp>(
          v.getLoc(), vreg_f32, src_row2, vreg_part);
      *tile = ctxt.builder.create<tpu::PackSubelementsOp>(
          v.getLoc(), src_row1.getType(), ValueRange{half_row1, half_row2});
    });
    src = new_src;
    src_tiles = std::move(src_tiles_retiled);
  }
  // (16, 128) -> (8, 128) tiling change for packed 16-bit types
  if (src.implicit_dim() != VectorLayout::ImplicitDim::kNone &&
      dst.implicit_dim() != VectorLayout::ImplicitDim::kNone &&
      vty.getElementType() == ctxt.builder.getBF16Type() &&
      src.tiling() == std::array<int64_t, 2>{16, 128} &&
      dst.tiling() == std::array<int64_t, 2>{8, 128}) {
    const VectorLayout new_src(src.bitwidth(), src.offsets(), dst.tiling());
    xla::Array<Value> src_tiles_retiled(
        new_src.tileArrayShape(vty.getShape(), ctxt.target_shape));
    src_tiles_retiled.Each([&](absl::Span<const int64_t> idx, Value *tile) {
      SmallVector<int64_t> src_idx(idx.begin(), idx.end());
      src_idx[src_idx.size() - 2] /= 2;
      src_idx[src_idx.size() - 1] *= 2;
      Value src_row1 = src_tiles(src_idx);
      if (src_idx[src_idx.size() - 1] + 1 <
          src_tiles.dim(src_tiles.num_dimensions() - 1)) {
        ++src_idx[src_idx.size() - 1];
      }
      Value src_row2 = src_tiles(src_idx);
      const int vreg_part = idx[idx.size() - 1] % 2;
      auto half_row1 = ctxt.builder.create<tpu::UnpackSubelementsOp>(
          v.getLoc(), vreg_f32, src_row1, vreg_part);
      auto half_row2 = ctxt.builder.create<tpu::UnpackSubelementsOp>(
          v.getLoc(), vreg_f32, src_row2, vreg_part);
      *tile = ctxt.builder.create<tpu::PackSubelementsOp>(
          v.getLoc(), src_row1.getType(), ValueRange{half_row1, half_row2});
    });
    src = new_src;
    src_tiles = std::move(src_tiles_retiled);
  }
  // TODO(kumudbhandari): Generalize the logic below to handle retiling from (8,
  // 128) to (x, 128) where x=1, 2 or 4.
  if (src.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      dst.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      src.offsets() == dst.offsets() && src.bitwidth() != 16 &&
      src.tiling() == std::array<int64_t, 2>{8, 128} &&
      dst.tiling() == std::array<int64_t, 2>{4, 128}) {
    const VectorLayout retiled_src_layout(src.bitwidth(), src.offsets(),
                                          dst.tiling());
    xla::Array<Value> retiled_src_tiles(
        retiled_src_layout.tileArrayShape(vty.getShape(), ctxt.target_shape));

    // Consider a value of type and shape: f32(8, 256). Retiling from (8,128) to
    // (4,128):
    // vreg (tile) array shape (1, 2), with original (8,128) tiling:
    //   vreg_0_0: slice:[0:7, 0:127] vreg_0_1: slice:[0:7, 128:255]

    // vreg (tile) array shape: (2, 1), with (4,128) retiling:
    //   vreg_0_0: slice: [0:3, 0:127], slice: [0:3, 128:255]
    //   vreg_1_0: slice:[4:7, 0:127], slice: [4:7, 128:255]
    const int64_t nd = retiled_src_tiles.num_dimensions();
    retiled_src_tiles.Each(
        [&](absl::Span<const int64_t> retiled_idx, Value *tile) {
          SmallVector<int64_t> src_tile_idx(toArrayRef(retiled_idx));

          // The first src tile, half of which forms the first half of the
          // retiled tile(retiled_row_idx, retiled_col_idx).
          src_tile_idx[nd - 2] /= 2;
          src_tile_idx[nd - 1] *= 2;
          const Value src_tile_1 = src_tiles(src_tile_idx);

          // The second src tile, half of which forms the second half of the
          // retiled tile(retiled_row_idx, retiled_col_idx).
          if (src_tile_idx[nd - 1] + 1 < src_tiles.dim(nd - 1)) {
            // TODO(tlongeri): is this just when the second tile is invalid? Can
            // we just set src_tile_2 to nullptr and not merge it in this
            // situation?
            ++src_tile_idx[nd - 1];
          }
          const Value src_tile_2 = src_tiles(src_tile_idx);

          // Each (retiled_row_idx)th tile is formed from 2 top or 2 bottom half
          // sublanes of the original tile.
          // We need to rotate sublanes of one of the two tiles to push either a
          // top half to the bottom or vice-versa.
          const Value tile_to_merge_1 =
              retiled_idx[nd - 2] % 2 == 0
                  ? src_tile_1
                  : ctxt.builder.create<tpu::RotateOp>(
                        v.getLoc(), src_tile_1, /*amount=*/4, /*dimension=*/0);
          const Value tile_to_merge_2 =
              retiled_idx[nd - 2] % 2 != 0
                  ? src_tile_2
                  : ctxt.builder.create<tpu::RotateOp>(
                        v.getLoc(), src_tile_2, /*amount=*/4, /*dimension=*/0);
          // Create a mask to select first half from tile 1 and second half of
          // data from tile 2 to be merged.
          const RectangularVregBounds vreg_half_bound(
              {0, 0}, {ctxt.target_shape[0] / 2, ctxt.target_shape[1]});
          const Value vreg_select_mask =
              vreg_half_bound
                  .getVectorMask(ctxt.builder, ctxt.hardware_generation,
                                 ctxt.target_shape)
                  .value();
          *tile = ctxt.builder.create<arith::SelectOp>(
              v.getLoc(), vreg_select_mask, tile_to_merge_1, tile_to_merge_2);
        });
    src = retiled_src_layout;
    src_tiles = std::move(retiled_src_tiles);
  }

  // Fix up the offsets, assuming everything else matches between src and dst.
  if (src.tiling() == dst.tiling() &&
      src.implicit_dim() == dst.implicit_dim()) {
    // TODO(apaszke): Changing an offset might add or remove one vreg.
    if (dst_tiles_shape != src_tiles.dimensions()) {
      return emitError(v.getLoc(), "Offsets changing the vreg array shape");
    }
    xla::Array<Value> dst_tiles = src_tiles;

    // Shifting rows
    int row_diff;
    if (!src.offsets()[0].has_value()) {
      row_diff = 0;
    } else if (!dst.offsets()[0].has_value()) {
      return emitError(v.getLoc(), "Sublane broadcast not implemented");
    } else {
      row_diff = *dst.offsets()[0] - *src.offsets()[0];
    }

    if (row_diff != 0) {
      // This is easy, because we never need to combine multiple vregs.
      const SmallVector<int64_t> implicit_shape =
          src.implicitShape(vty.getShape());
      if (implicit_shape[implicit_shape.size() - 2] != 1) {
        return emitError(v.getLoc(), "Row shifts for multi-row values");
      }
      CHECK(src.offsets()[0].has_value());
      CHECK(dst.offsets()[0].has_value());
      const int64_t src_sublane = *src.offsets()[0] / packing;
      const int64_t dst_sublane = *dst.offsets()[0] / packing;
      if (int64_t sublane_diff = dst_sublane - src_sublane) {
        if (sublane_diff < 0) {
          sublane_diff += ctxt.target_shape[0];
        }
        src_tiles.Each([&](absl::Span<const int64_t> idx, Value tile) {
          dst_tiles(idx) = ctxt.builder
                               .create<tpu::RotateOp>(v.getLoc(), tile,
                                                      /*amount=*/sublane_diff,
                                                      /*dimension=*/0)
                               .getResult();
        });
      }
      const int src_subelem = src_sublane % packing;
      const int dst_subelem = dst_sublane % packing;
      if (src_subelem != dst_subelem) {
        const int subelem_diff = dst_subelem - src_subelem;
        const int shift_bits = bitwidth * std::abs(subelem_diff);
        VectorType bits_vreg_ty =
            VectorType::get(ctxt.target_shape, ctxt.builder.getI32Type());
        auto shift_vreg = ctxt.builder.create<arith::ConstantOp>(
            v.getLoc(), bits_vreg_ty,
            DenseElementsAttr::get(bits_vreg_ty, shift_bits));
        absl::Status status = dst_tiles.EachStatus(
            [&](absl::Span<const int64_t> idx, Value tile) -> absl::Status {
              auto bit_tile = ctxt.builder.create<tpu::BitcastOp>(
                  v.getLoc(), bits_vreg_ty, tile);
              Operation *shift_tile;
              if (subelem_diff > 0) {
                shift_tile = ctxt.builder.create<arith::ShLIOp>(
                    v.getLoc(), bit_tile, shift_vreg);
              } else if (subelem_diff < 0) {
                shift_tile = ctxt.builder.create<arith::ShRUIOp>(
                    v.getLoc(), bit_tile, shift_vreg);
              } else {
                return xla::InvalidArgument("unexpected equal subelements");
              }
              dst_tiles(idx) = shift_tile->getResult(0);
              return absl::OkStatus();
            });
        if (!status.ok()) {
          return emitError(v.getLoc(), status.error_message());
        }
      }
    }
    int64_t col_diff;
    if (!src.offsets()[1].has_value()) {
      col_diff = 0;
    } else if (!dst.offsets()[1].has_value()) {
      return emitError(v.getLoc(),
                       "Not implemented: Lane broadcast not implemented");
    } else {
      col_diff = *dst.offsets()[1] - *src.offsets()[1];
    }
    if (col_diff != 0) {
      if (row_diff != 0) {
        return emitError(v.getLoc(),
                         "Not implemented: Both columns and rows are shifted");
      }
      if (col_diff < 0) {
        return emitError(v.getLoc(), "Not implemented: Shifts to the left");
      }
      if (bitwidth != 32) {
        return emitError(
            v.getLoc(), "Not implemented: Only 32-bit column shifts supported");
      }
      const int64_t sublane_diff = col_diff;
      CHECK_GE(src_tiles.num_dimensions(), 1);
      std::optional<tpu::CreateMaskOp> maybe_create_mask;
      if (src_tiles.dimensions()[src_tiles.num_dimensions() - 1] > 1) {
        auto boundIdxCst =
            std::bind(IdxCst, std::placeholders::_1, ctxt.builder, v.getLoc());
        maybe_create_mask = ctxt.builder.create<tpu::CreateMaskOp>(
            v.getLoc(),
            VectorType::get(ctxt.target_shape, ctxt.builder.getI1Type()),
            ValueRange{boundIdxCst(0), boundIdxCst(0)},
            ValueRange{boundIdxCst(ctxt.target_shape[0]),
                       boundIdxCst(col_diff)});
      }
      src_tiles.Each([&](absl::Span<const int64_t> idx, Value tile) {
        Value rot_tile = ctxt.builder
                             .create<tpu::RotateOp>(v.getLoc(), tile,
                                                    /*amount=*/sublane_diff,
                                                    /*dimension=*/1)
                             .getResult();
        if (idx[idx.size() - 1] != 0) {
          SmallVector<int64_t> prev_idx(idx.begin(), idx.end());
          --prev_idx[idx.size() - 1];
          Value prev_rot_tile = dst_tiles(prev_idx);
          rot_tile = ctxt.builder.create<arith::SelectOp>(
              v.getLoc(), maybe_create_mask->getResult(), prev_rot_tile, tile);
        }
        dst_tiles(idx) = rot_tile;
      });
    }
    return assemble(ctxt, vty, dst, std::move(dst_tiles)).getResult();
  }
  // TODO(apaszke): Implement general relayout
  return emitError(v.getLoc(), "unsupported layout change for ")
         << vty << ": " << src << " -> " << dst;
}

// Rewrites the operation according to its layout annotations.
//
// Args:
//   ctxt: The context used for rewriting.
//   op: An MLIR operation to be rewritten.
//
// A valid op is expected to have a layout_in attribute unless it has no
// operands. The layout_in attribute must fulfill the following:
//   - All vector operands originate from an operation (not a BlockArgument)
//   and
//     have a valid layout (Layout1D or Layout2D)
//   - All non-vector operands must have NoLayout.
// TODO(apaszke): Implement a debug mode that inserts additional assertions.
// For example, we should verify that ops that were supposed to generate
// replicated outputs satisfy that requirement.
LogicalResult applyLayoutOp(RewriteContext &ctxt, Operation &op) {
  ctxt.builder.setInsertionPointAfter(&op);
  // TODO(tlongeri): Once we remove the Python implementation, remove this.
  // This is to avoid failing checks due to the fact that unrolled ops have no
  // layout.
  if (!rules().contains(op.getName().getStringRef())) {
    return success();
  }
  CHECK(rules().find(op.getName().getStringRef()) != rules().end());

  // When an operation does not have any operands, the layout_in tuple is empty.
  // If one of the operands is a scalar value, the corresponding entry in the
  // layout_in tuple will be None. The same applies to the results of the
  // operation and the layout_out tuple.
  FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> layout_out,
                             getOutLayout(op));
  FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> layout_in,
                             getInLayout(op));
  if (!layout_in.empty()) {
    // Ensure that the out_layout at definition site matches the in_layout
    // declared here.
    for (auto [idx, tup] :
         llvm::enumerate(llvm::zip(op.getOperands(), layout_in))) {
      auto [operand, li] = tup;
      auto vty = dyn_cast<VectorType>(operand.getType());
      if ((vty == nullptr) == li.has_value()) {
        return op.emitError(
            "layout should be none iff operand is not a vector");
      }
      if (vty == nullptr) {
        continue;
      }

      // The operand should always be an Operation (and not a BlockArgument)
      // since we expect the FuncOp to have only memrefs and semaphores as
      // arguments.
      auto op_result = dyn_cast<OpResult>(operand);
      if (op_result == nullptr) {
        return op.emitError("expected operand to be an operation");
      }
      Operation *const def_op = op_result.getOwner();
      CHECK(def_op);
      const unsigned res_idx = op_result.getResultNumber();
      FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> def_layouts,
                                 getOutLayout(*def_op));
      const Layout lo = def_layouts[res_idx];
      if (!lo.has_value()) {
        return op.emitError() << "vector result should have a defined layout";
      }
      if (lo->generalizes(*li, std::nullopt, ctxt.target_shape)) {
        continue;
      }
      FAILUREOR_ASSIGN_OR_RETURN(Value new_v,
                                 relayout(ctxt, operand, /*src=*/*lo,
                                          /*dst=*/*li));
      op.setOperand(idx, new_v);
    }
  }

  const bool no_vector_args =
      llvm::none_of(layout_out,
                    [](Layout layout) { return layout.has_value(); }) &&
      llvm::none_of(layout_in,
                    [](Layout layout) { return layout.has_value(); });
  if (no_vector_args && op.getRegions().empty()) {
    // We don't need to do anything for scalar operations.
    if (!op.getOperands().empty()) {
      op.removeAttr("in_layout");
    }
    if (!op.getResults().empty()) {
      op.removeAttr("out_layout");
    }
    return success();
  }
  if (auto rule_it = rules().find(op.getName().getStringRef());
      rule_it != rules().end()) {
    rule_type rule = rule_it->getValue();
    // TODO(tlongeri): Support multiple outputs
    CHECK_EQ(layout_out.size(), 1);
    return rule(ctxt, op, layout_in, layout_out.front());
  }
  return op.emitError("Unsupported operation: ") << op.getName();
}

LogicalResult applyLayoutBlock(RewriteContext &ctxt, Block &block) {
  // We'll be modifying the block, so use early increment.
  for (Operation &op : make_early_inc_range(block)) {
    if (failed(applyLayoutOp(ctxt, op))) {
      return failure();
    }
  }
  return success();
}

// Rewrites the function according to layout annotations of its operations.
//
//   Args:
//     ctx: The context used for rewriting.
//     f: An MLIR function to be rewritten.
LogicalResult applyLayoutFunc(RewriteContext &ctxt, func::FuncOp f) {
  if (f->getNumRegions() != 1) {
    return f.emitError("Expected FuncOp to have a single region");
  }
  if (!f.getBody().hasOneBlock()) {
    return f.emitError("Expected FuncOp to have a single block");
  }
  return applyLayoutBlock(ctxt, f.getBody().front());
}

struct ApplyVectorLayoutPass
    : public impl::ApplyVectorLayoutPassBase<ApplyVectorLayoutPass> {
  ApplyVectorLayoutPass(int hardware_generation_, int lane_count_,
                        int sublane_count_) {
    hardware_generation = hardware_generation_;
    sublane_count = sublane_count_;
    lane_count = lane_count_;
  }
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder builder(func.getBody());
    RewriteContext ctxt{
        func, builder, hardware_generation, {sublane_count, lane_count}};
    if (failed(applyLayoutFunc(ctxt, func))) {
      signalPassFailure();
      return;
    }
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> createApplyVectorLayoutPass(
    int hardware_generation, int lane_count, int sublane_count) {
  return std::make_unique<ApplyVectorLayoutPass>(hardware_generation,
                                                 lane_count, sublane_count);
}

}  // namespace mlir::tpu
