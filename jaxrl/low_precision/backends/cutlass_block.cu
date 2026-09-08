#include <cuda_runtime.h>
#include <xla/ffi/api/ffi.h>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
namespace ffi=xla::ffi;
using namespace cute;
#ifndef BRC_FORMAT
#define BRC_FORMAT 0
#endif
#if BRC_FORMAT == 0
using Element=cutlass::mx_float8_t<cutlass::float_e4m3_t>;
constexpr int Alignment=16;
#elif BRC_FORMAT == 1
using Element=cutlass::nv_float4_t<cutlass::float_e2m1_t>;
constexpr int Alignment=32;
#elif BRC_FORMAT == 2
using Element=cutlass::mx_float4_t<cutlass::float_e2m1_t>;
constexpr int Alignment=32;
#endif
using Arch=cutlass::arch::Sm120;
using Op=cutlass::arch::OpClassBlockScaledTensorOp;
using BlockTile=Shape<_128,_128,_128>;
using Cluster=Shape<_1,_1,_1>;
using Epilogue=typename cutlass::epilogue::collective::CollectiveBuilder<Arch,Op,BlockTile,Cluster,
 cutlass::epilogue::collective::EpilogueTileAuto,float,float,
 float,cutlass::layout::RowMajor,4,float,cutlass::layout::RowMajor,4,
 cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using Mainloop=typename cutlass::gemm::collective::CollectiveBuilder<Arch,Op,
 Element,cutlass::layout::RowMajor,Alignment,Element,cutlass::layout::ColumnMajor,Alignment,float,BlockTile,Cluster,
 cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
 cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Kernel=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Mainloop,Epilogue,void>;
using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
#define CHECK(call) do {auto status=(call); if(status!=cutlass::Status::kSuccess) return ffi::Error::Internal(std::string(#call)+": "+cutlassGetStatusString(status));} while(0)
ffi::Error Matmul(cudaStream_t stream, ffi::Buffer<ffi::U8> x,ffi::Buffer<ffi::U8> w,
 ffi::Buffer<ffi::U8> sx,ffi::Buffer<ffi::U8> sw,ffi::ResultBuffer<ffi::F32> y,
 ffi::ResultBuffer<ffi::U8> workspace,int64_t m_,int64_t n_,int64_t k_) {
 int m=m_,n=n_,k=k_;
 using Config=typename Mainloop::Sm1xxBlkScaledConfig;
 auto sa=cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, {m,k,1});
 auto sb=cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, {n,k,1});
 auto sc=cutlass::make_cute_packed_stride(typename Kernel::StrideC{}, {m,n,1});
 auto sd=cutlass::make_cute_packed_stride(typename Kernel::StrideD{}, {m,n,1});
 auto la=Config::tile_atom_to_shape_SFA(make_shape(m,n,k,1));
 auto lb=Config::tile_atom_to_shape_SFB(make_shape(m,n,k,1));
 typename Gemm::Arguments args {cutlass::gemm::GemmUniversalMode::kGemm,{m,n,k,1},
  {reinterpret_cast<Element::DataType*>(x.typed_data()),sa,reinterpret_cast<Element::DataType*>(w.typed_data()),sb,
   reinterpret_cast<Element::ScaleFactorType*>(sx.typed_data()),la,reinterpret_cast<Element::ScaleFactorType*>(sw.typed_data()),lb},
  {{1.f,0.f},y->typed_data(),sc,y->typed_data(),sd}};
 if(Gemm::get_workspace_size(args)>workspace->element_count()) return ffi::Error::Internal("Insufficient XLA scratch");
 Gemm gemm;
 CHECK(gemm.can_implement(args));
 CHECK(gemm.initialize(args,workspace->typed_data(),stream));
 CHECK(gemm.run(stream));
 return ffi::Error::Success();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(brc_block,Matmul,ffi::Ffi::Bind()
 .Ctx<ffi::PlatformStream<cudaStream_t>>()
 .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
 .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
 .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::U8>>()
 .Attr<int64_t>("m").Attr<int64_t>("n").Attr<int64_t>("k"));
