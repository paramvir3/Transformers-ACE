#ifdef PAIR_CLASS

PairStyle(transformers_ace, PairTransformersACE);

#else

#ifndef LMP_PAIR_TRANSFORMERS_ACE_H
#define LMP_PAIR_TRANSFORMERS_ACE_H

#include "pair.h"

#include <torch/script.h>
#include <torch/torch.h>

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace LAMMPS_NS {

class PairTransformersACE : public Pair {
 public:
  PairTransformersACE(class LAMMPS *);
  ~PairTransformersACE() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;

 protected:
  void allocate();

 private:
  std::map<std::string, std::string> parse_metadata(const std::string &) const;
  std::vector<std::string> split_words(const std::string &) const;
  torch::Tensor cell_tensor() const;
  int local_rank() const;
  torch::Device resolve_device() const;

  torch::jit::script::Module model_;
  torch::Device device_ = torch::Device(torch::kCPU);
  std::string device_spec_ = "auto";
  bool model_loaded_ = false;
  double cutoff_ = 0.0;
  std::vector<int64_t> lammps_type_to_z_;
  std::vector<std::string> model_type_symbols_;
  std::vector<int64_t> model_type_atomic_numbers_;
};

}    // namespace LAMMPS_NS

#endif
#endif
