# Source before Cobot SFT / decode benchmarks so torchcodec finds FFmpeg
# and DataLoader workers do not oversubscribe BLAS/OpenMP threads.
#
#   source /data/gxy/realworldRL/RLinf/toolkits/lerobot/env_ffmpeg_torchcodec.sh

FFMPEG_PREFIX="${FFMPEG_PREFIX:-/data/gxy/realworldRL/ffmpeg_prefix}"
export PATH="${FFMPEG_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${FFMPEG_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# One BLAS/OpenMP thread per process so many DataLoader workers scale out.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Silence TensorFlow / oneDNN spam from DataLoader worker processes.
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-0}"
export GLOG_minloglevel="${GLOG_minloglevel:-3}"
export ABSL_MIN_LOG_LEVEL="${ABSL_MIN_LOG_LEVEL:-3}"
