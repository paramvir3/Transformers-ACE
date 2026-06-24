#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: ./patch_lammps.sh /path/to/lammps"
  exit 1
fi

lammps_dir="$1"
if [[ ! -d "$lammps_dir/src" || ! -f "$lammps_dir/cmake/CMakeLists.txt" ]]; then
  echo "$lammps_dir does not look like a LAMMPS source tree"
  exit 1
fi
if [[ ! -f pair_transformers_ace.cpp || ! -f pair_transformers_ace.h ]]; then
  echo "Run this script from the lammps/pair_style directory"
  exit 1
fi

echo "Copying pair_style transformers_ace into $lammps_dir/src"
cp pair_transformers_ace.cpp "$lammps_dir/src/"
cp pair_transformers_ace.h "$lammps_dir/src/"

python3 - "$lammps_dir/cmake/CMakeLists.txt" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
text = text.replace("set(CMAKE_CXX_STANDARD 11)", "set(CMAKE_CXX_STANDARD 17)")
block = """

message(STATUS "<< TRANSFORMERS_ACE flags >>")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(lammps PUBLIC "${TORCH_LIBRARIES}")
"""
if "<< TRANSFORMERS_ACE flags >>" not in text:
    text += block
path.write_text(text)
PY

echo "Done. Configure LAMMPS with:"
echo "  cmake ../cmake -DCMAKE_PREFIX_PATH=\$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
