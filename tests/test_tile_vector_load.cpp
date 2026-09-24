// Host-only contracts for vector LDS loads shared by HIP and Loom emitters.
#include "lse/backends/hrx/hipc/hip_types.hpp"
#include "lse/backends/hrx/hipc/hip_sources.hpp"
#include "lse/backends/hrx/loomc/loom_sources.hpp"
#include "lse/backends/hrx/loomc/loom_types.hpp"
#include "lse/backends/hrx/loomc/loom_print.hpp"
#include "lse/ir/alias.hpp"
#include "lse/ir/env.hpp"
#include "lse/ir/lower.hpp"
#include <cstdio>
#include <stdexcept>
#include <fstream>
namespace ir=lse::ir;
void need(bool v,const char *s){if(!v)throw std::runtime_error(s);}
template<class T> void test(const ir::TypeTable& types,bool loom,unsigned bytes,unsigned width){
 const auto intrinsics=loom?lse::backend::loom_sources():lse::backend::hip_sources();
 ir::KernelBody body(types,intrinsics,65536);ir::env::Emit e{&body};
 auto tile=e.lds<T>(64);need(bool(tile),"reserve");
 // Two nested aligned slices: effective offset is 8 + 8 + 8 = 24 elements.
 auto view=tile.slice(8,48).slice(8,32);
 const auto p=view.load(e.u32(8u),bytes);
 need(p.width()==width,"pack width");
 e.ret(p[0]);
 unsigned loads=0;
 body.ir().walk([&](ir::OpId id){const auto &op=body.ir().op(id);if(op.kind!=ir::OpKind::kLoadVec)return;
  ir::Access access;need(ir::access_of(body.ir(),id,&access),"alias recognizes load");
  need(access.space==ir::Space::kWorkgroup,"alias preserves shared space");
  need(access.buffer==tile.id() && !access.is_write && access.width==width,"alias range");
  need(access.index.as_constant()==24,"nested slice offset preserved");++loads;
 });need(loads==1,"exactly one vector load");
 if(loom){auto print=lse::backend::loom_print(body.ir(),{});need(print.ok(),"Loom lowering");
  need(print->text.find(width>1?"vector.load":"view.load")!=std::string::npos,"Loom load opcode");
  if(width==8)need(print->text.find("vector<8xbf16>")!=std::string::npos,"Loom BF16 element type");
  std::puts(print->text.c_str());
 }else{auto text=ir::lower(body.ir());need(!text.empty(),"HIP lowering");
  need(text.find("__shared__")!=std::string::npos,"HIP shared allocation");
  if(width>1)need(text.find("= *(const ")!=std::string::npos,"HIP typed vector pointer load");
  if(width==8)need(text.find("ext_vector_type(8)")!=std::string::npos,"HIP vector width");
  std::puts(text.c_str());
 }
}
int main(){try{
 for(bool loom:{false,true}){auto types=loom?lse::backend::loom_types():lse::backend::hip_types();
  test<ir::bf16>(types,loom,16,8);test<ir::bf16>(types,loom,8,4);
  test<ir::f32>(types,loom,16,4);test<ir::f32>(types,loom,4,1);
 }
 // Every lane's eight-element WMMA fragment remains contiguous and 16-byte
 // aligned after the existing K-major LDS bank rotation. Check all tiles.
 auto address=[](unsigned r,unsigned k){return r*64+((k/8+r%8)%8)*8+k%8;};
 for(unsigned r=0;r<64;++r)for(unsigned k=0;k<64;k+=8){
  auto base=address(r,k);need(base%8==0 && base+8<=4096,"fragment alignment/bounds");
  for(unsigned j=0;j<8;++j)need(address(r,k+j)==base+j,"fragment contiguity");
 }
 std::puts("PASS HIP/Loom BF16/F32 tile vector load, nested slices, shared alias ranges, all fragment alignments");return 0;
}catch(const std::exception &e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
