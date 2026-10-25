#include <gauge_field.h>
#include <color_spinor_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.hpp>
#include <kernels/dslash_wilson.cuh>

#if defined(QUDA_ENABLE_TC_COUNTER) && (QUDA_ENABLE_TC_COUNTER == 1)
#include <cstdio>
#endif

/**
   This is the basic gauged Wilson operator
   TODO
   - gauge fix support
*/

namespace quda
{

#if defined(QUDA_ENABLE_TC_COUNTER) && (QUDA_ENABLE_TC_COUNTER == 1)
  struct WilsonTcCounterTracker {
    unsigned long long last_count = 0ULL;
    unsigned long long policy_calls = 0ULL;
    ~WilsonTcCounterTracker()
    {
      if (last_count > 0ULL) {
        std::printf("QUDA_TC_HMMA_EXECUTED=%llu (dslash_policy_calls=%llu)\n", last_count, policy_calls);
        std::fflush(stdout);
      }
    }
  };
#endif

  template <typename Arg> class Wilson : public Dslash<wilson, Arg>
  {
    using Dslash = Dslash<wilson, Arg>;
    const GaugeField &U;

  public:
    Wilson(Arg &arg, const GaugeField &U, cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
           const ColorSpinorField &halo) :
      Dslash(arg, out, in, halo), U(U)
    {
    }

    unsigned int maxBlockSize(const TuneParam &param) const override
    {
#if defined(QUDA_WILSON_MMA_SMEM) && (QUDA_WILSON_MMA_SMEM == 1)
      if constexpr (sizeof(typename mapper<typename Arg::Float>::type) == sizeof(float) && Arg::nColor == 3) {
        #if QUDA_WILSON_MMA_MODE == 8
        return 256 / (param.block.y * param.block.z);
#else
        return 128 / (param.block.y * param.block.z);
#endif
      }
#endif
      return Dslash::maxBlockSize(param);
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp, U);
      Dslash::template instantiate<packShmem>(tp, stream);
    }
  };

  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon> struct WilsonApply {

    template <bool distance_pc>
    WilsonApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                cvector_ref<const ColorSpinorField> &x, const GaugeField &U, double a, double alpha0, int t0,
                int parity, bool dagger, const int *comm_override, DistanceType<distance_pc>, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      auto halo = ColorSpinorField::create_comms_batch(in);

      WilsonArg<Float, nColor, nDim, DDArg, recon, distance_pc> arg(out, in, halo, U, a, x, parity, dagger,
                                                                    comm_override, alpha0, t0);
      Wilson<decltype(arg)> wilson(arg, U, out, in, halo);
      dslash::DslashPolicyTune<decltype(wilson)> policy(wilson, out, in, halo, profile);
#if defined(QUDA_ENABLE_TC_COUNTER) && (QUDA_ENABLE_TC_COUNTER == 1)
      if constexpr (sizeof(typename mapper<Float>::type) == sizeof(float) && nColor == 3) {
        static WilsonTcCounterTracker tracker;
        tracker.policy_calls++;
        unsigned long long h_cnt = 0ULL;
        cudaMemcpyFromSymbol(&h_cnt, quda_tc_hmma_counter_d, sizeof(unsigned long long));
        tracker.last_count = h_cnt;
        if (tracker.policy_calls == 1ULL || (tracker.policy_calls % 5000ULL) == 0ULL) {
          std::printf("QUDA_TC_HMMA_EXECUTED=%llu (dslash_policy_calls=%llu)\n", tracker.last_count,
                      tracker.policy_calls);
          std::fflush(stdout);
        }
      }
#endif
    }
  };

} // namespace quda
