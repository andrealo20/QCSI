/**
 * Cost and footprint of one pipeline, reported per frame.
 *
 * Operation counts are exact and architecture-independent; they are what CI
 * can watch for regressions. They are NOT cycle counts. Real cycle figures
 * need a cross-compiler and either Renode or hardware, neither of which this
 * measurement claims to substitute for.
 */
#include "qcsi/pipeline.h"
#include "qdsp/profile.h"
#include <stdio.h>
#include <math.h>

#define NSUB 30
#define NANT 3
#define NFRAMES 32
#define NFFT 32
#define NDOP 8

static qcsi_pipeline pipe;
static qdsp_cplx_q15 frame[NSUB*NANT];

int main(void){
    qcsi_pipeline_config cfg = {NSUB,NFRAMES,NFFT,NDOP,0,1,NANT};
    qdsp_profile_t p;
    int t,a,k;

    if (qcsi_pipeline_init(&pipe,&cfg,NULL)!=QDSP_OK){puts("init failed");return 1;}

    qdsp_profile_reset();
    for(t=0;t<NFRAMES;++t){
        for(a=0;a<NANT;++a) for(k=0;k<NSUB;++k){
            double ph=0.3*sin(0.2*k)+2.0*sin(0.31*t);
            frame[a*NSUB+k].re=qdsp_f32_to_q15((float)(0.45*cos(ph)));
            frame[a*NSUB+k].im=qdsp_f32_to_q15((float)(0.45*sin(ph)));
        }
        qcsi_pipeline_push(&pipe,frame);
    }
    p=qdsp_profile_get();

    printf("# qcsi - cost and footprint\n\n");
    printf("Configuration: %d subcarriers, %d antennas, %d-frame window, "
           "%d-point Doppler FFT, %d bins kept.\n\n",NSUB,NANT,NFRAMES,NFFT,NDOP);

    printf("## Static footprint\n\n");
    printf("| Buffer | Bytes |\n|---|---|\n");
    printf("| amplitude window | %lu |\n",(unsigned long)(sizeof(q15_t)*NFRAMES*NSUB));
    printf("| phase accumulators | %lu |\n",
           (unsigned long)((sizeof(int32_t)+sizeof(int64_t)+sizeof(int32_t)+sizeof(qcsi_angle_t))*NSUB));
    printf("| FFT scratch | %lu |\n",(unsigned long)(sizeof(qdsp_cplx_q15)*NFFT));
    printf("| features + scores | %lu |\n",
           (unsigned long)(sizeof(q15_t)*QCSI_MAX_FEATURES+sizeof(q63_t)*QCSI_MAX_CLASSES));
    printf("| **context total** | **%lu** (%.1f KiB) |\n\n",
           (unsigned long)qcsi_pipeline_footprint(),
           qcsi_pipeline_footprint()/1024.0);
    printf("Compile-time limits are generous (%d subcarriers, %d-frame window). "
           "Sized exactly for this configuration the context would be far "
           "smaller; the limits are what make the struct static rather than "
           "allocated.\n\n",QCSI_MAX_SUBCARRIERS,QCSI_MAX_WINDOW);

    printf("## Cost per frame\n\n");
    if(!QDSP_PROFILE_ENABLED){
        printf("> Operation counting is off: rebuild with -DQCSI_PROFILE=ON.\n");
        return 0;
    }
    printf("| Operation | Per window | Per frame |\n|---|---|---|\n");
    printf("| multiply-accumulate | %llu | %.1f |\n",
           (unsigned long long)p.mac,(double)p.mac/NFRAMES);
    printf("| rounding multiply | %llu | %.1f |\n",
           (unsigned long long)p.mul,(double)p.mul/NFRAMES);
    printf("| accumulator to q15 | %llu | %.1f |\n",
           (unsigned long long)p.round,(double)p.round/NFRAMES);
    printf("\nExact and architecture-independent. Not cycle counts.\n");
    return 0;
}
