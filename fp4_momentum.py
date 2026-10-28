"""Two FP4 limbs for eight Lie algebra coefficients: 10 bytes/link."""
import torch
import triton
import triton.language as tl
from fp4_color import _e2m1, _decode

@triton.jit
def _pack8(X,Q,S,R:tl.constexpr,BLOCK:tl.constexpr):
    r=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
    j=tl.arange(0,4)
    a=tl.load(X+r[:,None]*8+2*j[None,:],(r[:,None]<R)&(j[None,:]<4),0).to(tl.float32)
    b=tl.load(X+r[:,None]*8+2*j[None,:]+1,(r[:,None]<R)&(j[None,:]<4),0).to(tl.float32)
    scale=tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.maximum(tl.abs(a),tl.abs(b)),1)/6.,2.**-9))))
    q=_e2m1(a/scale[:,None])|(_e2m1(b/scale[:,None])<<4)
    tl.store(Q+r[:,None]*4+j[None,:],q,(r[:,None]<R)&(j[None,:]<4))
    tl.store(S+r,scale,r<R)


@triton.jit
def _unpack8(Q,S,Y,R:tl.constexpr,BLOCK:tl.constexpr):
    r=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
    j=tl.arange(0,4)
    q=tl.load(Q+r[:,None]*4+j[None,:],(r[:,None]<R)&(j[None,:]<4),0)
    s=tl.load(S+r,r<R,0.).to(tl.float32)
    tl.store(Y+r[:,None]*8+2*j[None,:],_decode(q&15)*s[:,None],(r[:,None]<R)&(j[None,:]<4))
    tl.store(Y+r[:,None]*8+2*j[None,:]+1,_decode(q>>4)*s[:,None],(r[:,None]<R)&(j[None,:]<4))


def encode8(x):
    x=x.contiguous();r=x.numel()//8
    q=torch.empty((*x.shape[:-1],4),device=x.device,dtype=torch.uint8)
    s=torch.empty(x.shape[:-1],device=x.device,dtype=torch.float8_e4m3fn)
    _pack8[(triton.cdiv(r,128),)](x,q,s,r,128)
    if not torch.isfinite(s.float()).all():raise ValueError('Momentum scale overflow')
    return q,s

def decode8(q,s):
    q=q.contiguous();s=s.contiguous()
    x=torch.empty((*s.shape,8),device=q.device)
    _unpack8[(triton.cdiv(s.numel(),128),)](q,s,x,s.numel(),128)
    return x

class CompactMomentum:
    def __init__(self,shape):
        self.shape=shape
        prefix=shape[:-2]
        self.q=torch.empty((*prefix,4),device='cuda',dtype=torch.uint8)
        self.s=torch.empty(prefix,device='cuda',dtype=torch.float8_e4m3fn)
        self.lq=torch.empty_like(self.q);self.ls=torch.empty_like(self.s)
    def put(self,index,value):
        x=value.flatten(-2)[...,:8].contiguous()
        q,s=encode8(x);lq,ls=encode8((x-decode8(q,s))*256.)
        self.q[index]=q;self.s[index]=s;self.lq[index]=lq;self.ls[index]=ls
    def get(self,index=...):
        x=decode8(self.q[index],self.s[index])+decode8(self.lq[index],self.ls[index])/256.
        padded=torch.zeros((*x.shape[:-1],12),device=x.device)
        padded[...,:8]=x
        return padded.reshape(*x.shape[:-1],2,6)
    def bytes(self):return sum(t.numel()*t.element_size() for t in (self.q,self.s,self.lq,self.ls))
