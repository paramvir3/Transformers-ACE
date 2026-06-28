#include "pair_transformers_ace.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace LAMMPS_NS;

PairTransformersACE::PairTransformersACE(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
}

PairTransformersACE::~PairTransformersACE()
{
  if (copymode) return;
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

void PairTransformersACE::allocate()
{
  allocated = 1;
  int n = atom->ntypes;
  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
}

void PairTransformersACE::settings(int narg, char **arg)
{
  if (narg == 0) return;
  if (narg == 2 && strcmp(arg[0], "device") == 0) {
    device_spec_ = arg[1];
    return;
  }
  error->all(FLERR, "Illegal pair_style transformers_ace command. Use: pair_style transformers_ace [device auto|cpu|cuda|cuda:N]");
}

std::vector<std::string> PairTransformersACE::split_words(const std::string &line) const
{
  std::stringstream stream(line);
  std::vector<std::string> words;
  std::string word;
  while (stream >> word) words.push_back(word);
  return words;
}

std::map<std::string, std::string> PairTransformersACE::parse_metadata(const std::string &text) const
{
  std::map<std::string, std::string> metadata;
  std::stringstream stream(text);
  std::string line;
  while (std::getline(stream, line)) {
    auto pos = line.find('=');
    if (pos == std::string::npos) continue;
    metadata[line.substr(0, pos)] = line.substr(pos + 1);
  }
  return metadata;
}

int PairTransformersACE::local_rank() const
{
#if MPI_VERSION >= 3
  MPI_Comm node_comm;
  if (MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, comm->me, MPI_INFO_NULL, &node_comm) ==
      MPI_SUCCESS) {
    int rank = 0;
    MPI_Comm_rank(node_comm, &rank);
    MPI_Comm_free(&node_comm);
    return rank;
  }
#endif

  const char *names[] = {
      "LOCAL_RANK",
      "OMPI_COMM_WORLD_LOCAL_RANK",
      "MV2_COMM_WORLD_LOCAL_RANK",
      "SLURM_LOCALID",
      "MPI_LOCALRANKID",
      nullptr,
  };
  for (const char **name = names; *name != nullptr; ++name) {
    const char *value = std::getenv(*name);
    if (value == nullptr || value[0] == '\0') continue;
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end != value && parsed >= 0) return static_cast<int>(parsed);
  }
  return comm->me;
}

torch::Device PairTransformersACE::resolve_device() const
{
  if (device_spec_ == "cpu") return torch::Device(torch::kCPU);

  if (device_spec_.rfind("cuda", 0) == 0) {
    if (!torch::cuda::is_available()) {
      error->all(FLERR, "pair_style transformers_ace requested CUDA but LibTorch reports no CUDA device");
    }
    const int count = static_cast<int>(torch::cuda::device_count());
    if (count <= 0) {
      error->all(FLERR, "pair_style transformers_ace requested CUDA but no CUDA devices are visible");
    }
    int index = local_rank() % count;
    const auto colon = device_spec_.find(':');
    if (colon != std::string::npos) {
      const std::string index_text = device_spec_.substr(colon + 1);
      char *end = nullptr;
      const long parsed = std::strtol(index_text.c_str(), &end, 10);
      if (end == index_text.c_str() || *end != '\0') {
        error->all(FLERR, "pair_style transformers_ace CUDA device must be written as cuda:N");
      }
      index = static_cast<int>(parsed);
      if (index < 0 || index >= count) {
        error->all(FLERR, "pair_style transformers_ace CUDA device index is outside the visible device range");
      }
    }
    return torch::Device(torch::kCUDA, index);
  }

  if (device_spec_ == "auto") {
    if (torch::cuda::is_available() && torch::cuda::device_count() > 0) {
      const int count = static_cast<int>(torch::cuda::device_count());
      return torch::Device(torch::kCUDA, local_rank() % count);
    }
    return torch::Device(torch::kCPU);
  }

  error->all(FLERR, "pair_style transformers_ace device must be auto, cpu, cuda, or cuda:N");
  return torch::Device(torch::kCPU);
}

void PairTransformersACE::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  const int ntypes = atom->ntypes;
  if (narg != 3 + ntypes) {
    error->all(FLERR, "pair_coeff must be: * * model.transformers_ace.pt type1 type2 ...");
  }
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0) {
    error->all(FLERR, "pair_coeff transformers_ace requires leading '* *'");
  }

  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++) setflag[i][j] = 0;

  std::string model_path(arg[2]);
  device_ = resolve_device();
  torch::jit::ExtraFilesMap extra_files;
  extra_files["metadata.txt"] = "";
  try {
    model_ = torch::jit::load(model_path, device_, extra_files);
    model_.eval();
  } catch (const c10::Error &err) {
    std::string message = "Could not load Transformers-ACE TorchScript model: " + model_path;
    error->all(FLERR, message.c_str());
  }
  model_loaded_ = true;

  const auto metadata = parse_metadata(extra_files["metadata.txt"]);
  if (!metadata.count("r_max") || !metadata.count("type_symbols") ||
      !metadata.count("type_atomic_numbers")) {
    error->all(FLERR, "Transformers-ACE model is missing metadata.txt fields");
  }
  cutoff_ = std::stod(metadata.at("r_max"));
  model_type_symbols_ = split_words(metadata.at("type_symbols"));
  auto z_words = split_words(metadata.at("type_atomic_numbers"));
  if (model_type_symbols_.size() != z_words.size()) {
    error->all(FLERR, "Transformers-ACE metadata type_symbols/type_atomic_numbers mismatch");
  }
  model_type_atomic_numbers_.clear();
  for (const auto &word : z_words) model_type_atomic_numbers_.push_back(std::stoll(word));

  lammps_type_to_z_.assign(ntypes + 1, -1);
  if (comm->me == 0) {
    std::cout << "Transformers-ACE: loading " << model_path << " on " << device_ << "\n";
    std::cout << "Transformers-ACE type mapping:\n";
  }
  for (int itype = 1; itype <= ntypes; itype++) {
    const std::string requested(arg[2 + itype]);
    auto found = std::find(model_type_symbols_.begin(), model_type_symbols_.end(), requested);
    if (found == model_type_symbols_.end()) {
      std::string message = "LAMMPS type name not found in Transformers-ACE model metadata: " + requested;
      error->all(FLERR, message.c_str());
    }
    const int model_idx = static_cast<int>(std::distance(model_type_symbols_.begin(), found));
    lammps_type_to_z_[itype] = model_type_atomic_numbers_[model_idx];
    if (comm->me == 0) {
      std::cout << "  LAMMPS type " << itype << " -> " << requested
                << " (Z=" << lammps_type_to_z_[itype] << ")\n";
    }
  }

  for (int i = 1; i <= ntypes; i++) {
    for (int j = i; j <= ntypes; j++) {
      if (lammps_type_to_z_[i] > 0 && lammps_type_to_z_[j] > 0) setflag[i][j] = 1;
    }
  }
}

void PairTransformersACE::init_style()
{
  if (!model_loaded_) error->all(FLERR, "pair_coeff must be set before pair_style transformers_ace");
  if (comm->nprocs > 1 && !force->newton_pair) {
    error->all(FLERR, "parallel pair_style transformers_ace requires 'newton on' so ghost-atom forces are reverse-communicated");
  }
  if (atom->tag_enable == 0) error->all(FLERR, "pair_style transformers_ace requires atom IDs");
  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
}

double PairTransformersACE::init_one(int, int)
{
  return cutoff_;
}

torch::Tensor PairTransformersACE::cell_tensor() const
{
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor cell = torch::zeros({3, 3}, options);
  auto c = cell.accessor<float, 2>();
  c[0][0] = static_cast<float>(domain->xprd);
  c[1][0] = static_cast<float>(domain->xy);
  c[1][1] = static_cast<float>(domain->yprd);
  c[2][0] = static_cast<float>(domain->xz);
  c[2][1] = static_cast<float>(domain->yz);
  c[2][2] = static_cast<float>(domain->zprd);
  return cell;
}

void PairTransformersACE::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  const int nlocal = atom->nlocal;
  const int nall = atom->nlocal + atom->nghost;
  if (nlocal <= 0) return;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  tagint *tag = atom->tag;

  std::vector<int> force_owner;
  if (comm->nprocs == 1) {
    std::unordered_map<tagint, int> local_by_tag;
    local_by_tag.reserve(nlocal);
    for (int i = 0; i < nlocal; i++) local_by_tag[tag[i]] = i;

    force_owner.assign(nall, -1);
    for (int i = 0; i < nlocal; i++) force_owner[i] = i;
    for (int i = nlocal; i < nall; i++) {
      auto found = local_by_tag.find(tag[i]);
      if (found != local_by_tag.end()) force_owner[i] = found->second;
    }
  }

  std::vector<int64_t> senders;
  std::vector<int64_t> receivers;
  senders.reserve(nlocal * 32);
  receivers.reserve(nlocal * 32);

  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  const double cutsq_model = cutoff_ * cutoff_;

  for (int ii = 0; ii < list->inum; ii++) {
    int i = ilist[ii];
    int *jlist = firstneigh[i];
    for (int jj = 0; jj < numneigh[i]; jj++) {
      int j = jlist[jj] & NEIGHMASK;
      const double dx = x[j][0] - x[i][0];
      const double dy = x[j][1] - x[i][1];
      const double dz = x[j][2] - x[i][2];
      const double rsq = dx * dx + dy * dy + dz * dz;
      if (rsq <= cutsq_model && rsq > 0.0) {
        senders.push_back(j);
        receivers.push_back(i);
      }
    }
  }

  auto fopts = torch::TensorOptions().dtype(torch::kFloat32);
  auto iopts = torch::TensorOptions().dtype(torch::kInt64);

  torch::Tensor z_tensor = torch::empty({nall}, iopts);
  torch::Tensor pos_tensor = torch::empty({nall, 3}, fopts);
  torch::Tensor local_mask = torch::zeros({nall}, fopts);
  auto z_acc = z_tensor.accessor<int64_t, 1>();
  auto pos_acc = pos_tensor.accessor<float, 2>();
  auto mask_acc = local_mask.accessor<float, 1>();
  for (int i = 0; i < nall; i++) {
    z_acc[i] = lammps_type_to_z_[type[i]];
    pos_acc[i][0] = static_cast<float>(x[i][0]);
    pos_acc[i][1] = static_cast<float>(x[i][1]);
    pos_acc[i][2] = static_cast<float>(x[i][2]);
    mask_acc[i] = (i < nlocal) ? 1.0f : 0.0f;
  }

  const int64_t nedges = static_cast<int64_t>(senders.size());
  torch::Tensor edge_index = torch::empty({2, nedges}, iopts);
  auto edge_acc = edge_index.accessor<int64_t, 2>();
  for (int64_t e = 0; e < nedges; e++) {
    edge_acc[0][e] = senders[e];
    edge_acc[1][e] = receivers[e];
  }
  torch::Tensor edge_shift = torch::zeros({nedges, 3}, fopts);
  torch::Tensor cell = cell_tensor();
  torch::Tensor strain = torch::zeros({6}, fopts);

  pos_tensor = pos_tensor.to(device_);
  pos_tensor.set_requires_grad(true);
  strain = strain.to(device_);
  strain.set_requires_grad(true);

  std::vector<torch::jit::IValue> inputs;
  inputs.emplace_back(z_tensor.to(device_));
  inputs.emplace_back(pos_tensor);
  inputs.emplace_back(cell.to(device_));
  inputs.emplace_back(edge_index.to(device_));
  inputs.emplace_back(edge_shift.to(device_));
  inputs.emplace_back(strain);
  inputs.emplace_back(local_mask.to(device_));

  torch::AutoGradMode enable_grad(true);
  torch::Tensor energy = model_.forward(inputs).toTensor();
  std::vector<torch::Tensor> grad_outputs = {energy};
  std::vector<torch::Tensor> grad_inputs = {pos_tensor, strain};
  auto grads = torch::autograd::grad(grad_outputs, grad_inputs);
  torch::Tensor forces = (-grads[0]).to(torch::kCPU);
  torch::Tensor strain_grad = grads[1].to(torch::kCPU);

  auto force_acc = forces.accessor<float, 2>();
  if (comm->nprocs == 1) {
    for (int i = 0; i < nall; i++) {
      const int owner = force_owner[i];
      if (owner < 0) continue;
      f[owner][0] += force_acc[i][0];
      f[owner][1] += force_acc[i][1];
      f[owner][2] += force_acc[i][2];
    }
  } else {
    for (int i = 0; i < nall; i++) {
      f[i][0] += force_acc[i][0];
      f[i][1] += force_acc[i][1];
      f[i][2] += force_acc[i][2];
    }
  }

  eng_vdwl = energy.detach().to(torch::kCPU).item<double>();

  if (vflag) {
    auto g = strain_grad.accessor<float, 1>();
    virial[0] += -g[0];
    virial[1] += -g[1];
    virial[2] += -g[2];
    virial[3] += -0.5 * g[3];
    virial[4] += -0.5 * g[4];
    virial[5] += -0.5 * g[5];
  }
  if (vflag_atom) {
    error->all(FLERR, "pair_style transformers_ace does not support per-atom virial yet");
  }
}
