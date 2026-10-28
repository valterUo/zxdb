// Scoped in-process pivot, local complementation and gadget pivot for Memgraph.
// Each procedure selects independent matches and rewrites to a fixed point.
// The older zero-argument gadget procedure remains for comparison.
#include <mgp.hpp>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <utility>
#include <vector>
#include <string>
#include <stdexcept>
#include <unordered_set>

namespace {
constexpr double EPS = 1e-9;

double fmodpos(double x, double m) {
  double r = std::fmod(x, m);
  if (r < 0) r += m;
  return r;
}
double getPhase(const mgp::Node &v) {
  auto p = v.GetProperty("phase");
  return (!p.IsNull() && p.IsNumeric()) ? p.ValueNumeric() : 0.0;
}
int64_t getT(const mgp::Node &v) {
  auto p = v.GetProperty("t");
  return (!p.IsNull() && p.IsNumeric()) ? static_cast<int64_t>(p.ValueNumeric()) : -1;
}
int64_t relT(const mgp::Relationship &r) {
  auto p = r.GetProperty("t");
  return (!p.IsNull() && p.IsNumeric()) ? static_cast<int64_t>(p.ValueNumeric()) : -1;
}
bool isPauli(double p) {
  double r = fmodpos(p, 2.0);
  return std::fabs(r) < EPS || std::fabs(r - 1.0) < EPS || std::fabs(r - 2.0) < EPS;
}
int64_t degree(const mgp::Node &v) {
  int64_t d = 0;
  for (auto r : v.OutRelationships()) { (void)r; ++d; }
  for (auto r : v.InRelationships()) { (void)r; ++d; }
  return d;
}
bool interior(const mgp::Node &v) {
  for (auto r : v.OutRelationships())
    if (relT(r) != 2 || getT(r.To()) != 1) return false;
  for (auto r : v.InRelationships())
    if (relT(r) != 2 || getT(r.From()) != 1) return false;
  return true;
}
using NbrMap = std::map<int64_t, std::pair<mgp::Relationship, mgp::Node>>;
NbrMap nbrs(const mgp::Node &v) {
  NbrMap m;
  for (auto r : v.OutRelationships()) m.insert({r.To().Id().AsInt(), {r, r.To()}});
  for (auto r : v.InRelationships()) m.insert({r.From().Id().AsInt(), {r, r.From()}});
  return m;
}
void addH(mgp::Graph &g, const mgp::Node &u, const mgp::Node &w, const mgp::Value &gid) {
  auto e = g.CreateRelationship(u, w, "Wire");
  e.SetProperty("t", mgp::Value(static_cast<int64_t>(2)));
  e.SetProperty("graph_id", gid);
}
void toggle(mgp::Graph &g, const std::vector<mgp::Node> &A,
            const std::vector<mgp::Node> &B, const mgp::Value &gid) {
  for (const auto &a : A) {
    std::map<int64_t, mgp::Relationship> anb;
    for (auto r : a.OutRelationships()) anb.insert({r.To().Id().AsInt(), r});
    for (auto r : a.InRelationships()) anb.insert({r.From().Id().AsInt(), r});
    for (const auto &b : B) {
      auto it = anb.find(b.Id().AsInt());
      if (it != anb.end() && relT(it->second) == 2) g.DeleteRelationship(it->second);
      // On Z spiders, adding H beside a simple edge retains the simple edge
      // and contributes pi. Never delete an existing simple edge here.
      else if (it != anb.end()) {
        auto node = a;
        node.SetProperty("phase", mgp::Value(fmodpos(getPhase(a) + 1.0, 2.0)));
      }
      else addH(g, a, b, gid);
    }
  }
}
void apply(mgp::Graph &g, mgp::Node &zj, mgp::Node &za, NbrMap &zjn, NbrMap &zan, bool keepJ = false) {
  auto gid = zj.GetProperty("graph_id");
  double pj = getPhase(zj), pa = getPhase(za);
  int64_t zjid = zj.Id().AsInt(), zaid = za.Id().AsInt();
  std::set<int64_t> jset, aset;
  for (auto &kv : zjn) if (kv.first != zaid) jset.insert(kv.first);
  for (auto &kv : zan) if (kv.first != zjid) aset.insert(kv.first);
  std::vector<mgp::Node> Lj, La, Ls;
  for (auto &kv : zjn) {
    if (kv.first == zaid) continue;
    if (aset.count(kv.first)) Ls.push_back(kv.second.second);
    else Lj.push_back(kv.second.second);
  }
  for (auto &kv : zan) {
    if (kv.first == zjid) continue;
    if (!jset.count(kv.first)) La.push_back(kv.second.second);
  }
  toggle(g, Lj, La, gid);
  toggle(g, Lj, Ls, gid);
  toggle(g, La, Ls, gid);
  for (auto &v : La) v.SetProperty("phase", mgp::Value(fmodpos(getPhase(v) + pj, 2.0)));
  for (auto &v : Ls) v.SetProperty("phase", mgp::Value(fmodpos(getPhase(v) + pj + 1.0, 2.0)));
  auto axis = g.CreateNode();
  axis.AddLabel("Node");
  axis.SetProperty("t", mgp::Value(static_cast<int64_t>(1)));
  axis.SetProperty("phase", mgp::Value(fmodpos(pj, 2.0)));
  axis.SetProperty("graph_id", gid);
  auto tip = g.CreateNode();
  tip.AddLabel("Node");
  tip.SetProperty("t", mgp::Value(static_cast<int64_t>(1)));
  tip.SetProperty("phase", mgp::Value(fmodpos(fmodpos(pa, 2.0) + 2.0, 2.0)));
  tip.SetProperty("graph_id", gid);
  addH(g, axis, tip, gid);
  for (auto &v : Lj) addH(g, axis, v, gid);
  for (auto &v : Ls) addH(g, axis, v, gid);
  if (!keepJ) g.DetachDeleteNode(zj);
  g.DetachDeleteNode(za);
}
bool owned(const mgp::Node &n, const std::string &gid) {
  auto value = n.GetProperty("graph_id");
  return value.IsString() && value.ValueString() == gid;
}

// Validate before any mutation, including direct calls that bypass Python.
void validateScope(mgp::Graph &g, const std::string &gid) {
  for (auto n : g.Nodes()) {
    if (!owned(n, gid)) continue;
    std::unordered_set<int64_t> seen;
    auto check = [&](const mgp::Relationship &r, const mgp::Node &other) {
      if (!owned(other, gid)) throw std::invalid_argument("Cross-graph relationship");
      if (other.Id() == n.Id() || !seen.insert(other.Id().AsInt()).second)
        throw std::invalid_argument("Native rewrites require normalized simple graphs");
      if (relT(r) != 1 && relT(r) != 2)
        throw std::invalid_argument("Unsupported edge type");
    };
    for (auto r : n.OutRelationships()) check(r, r.To());
    for (auto r : n.InRelationships()) check(r, r.From());
  }
}

int64_t onePass(mgp::Graph &g, const std::string *gid = nullptr) {
  std::set<int64_t> consumed;
  int64_t applied = 0;
  std::vector<std::pair<int64_t, mgp::Node>> nodes;
  for (auto n : g.Nodes())
    if (!gid || owned(n, *gid)) nodes.emplace_back(n.Id().AsInt(), n);
  std::sort(nodes.begin(), nodes.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
  for (auto &[zjid, zj] : nodes) {
    if (consumed.count(zjid)) continue;  // Do not dereference deleted handles.
    if (getT(zj) != 1) continue;
    double pj = getPhase(zj);
    if (!isPauli(pj) || !interior(zj)) continue;
    auto zjn = nbrs(zj);
    bool isAxis = false;
    for (auto &kv : zjn) if (degree(kv.second.second) == 1) { isAxis = true; break; }
    if (isAxis) continue;
    for (auto &kv : zjn) {           // std::map -> ascending id == min-id z_alpha first
      mgp::Node za = kv.second.second;
      int64_t aid = kv.first;
      if (consumed.count(aid) || getT(za) != 1) continue;
      double pa = getPhase(za);
      if (isPauli(pa) || degree(za) <= 1 || !interior(za)) continue;
      auto zan = nbrs(za);
      std::set<int64_t> claim{zjid, aid};
      for (auto &k2 : zjn) claim.insert(k2.first);
      for (auto &k2 : zan) claim.insert(k2.first);
      bool overlap = false;
      for (auto c : claim) if (consumed.count(c)) { overlap = true; break; }
      if (overlap) continue;
      apply(g, zj, za, zjn, zan);
      for (auto c : claim) consumed.insert(c);
      ++applied;
      break;
    }
  }
  return applied;
}

void applyPivot(mgp::Graph &g, mgp::Node a, mgp::Node b,
                const NbrMap &an, const NbrMap &bn, bool keepA = false) {
  std::vector<mgp::Node> left, right, shared;
  for (auto &[id, item] : an) {
    if (id == b.Id().AsInt()) continue;
    (bn.count(id) ? shared : left).push_back(item.second);
  }
  for (auto &[id, item] : bn)
    if (id != a.Id().AsInt() && !an.count(id)) right.push_back(item.second);
  auto gid = a.GetProperty("graph_id");
  double pa = getPhase(a), pb = getPhase(b);
  toggle(g, left, right, gid);
  toggle(g, left, shared, gid);
  toggle(g, right, shared, gid);
  for (auto n : left) n.SetProperty("phase", mgp::Value(fmodpos(getPhase(n) + pb, 2.0)));
  for (auto n : right) n.SetProperty("phase", mgp::Value(fmodpos(getPhase(n) + pa, 2.0)));
  for (auto n : shared) n.SetProperty("phase", mgp::Value(fmodpos(getPhase(n) + pa + pb + 1, 2.0)));
  if (!keepA) g.DetachDeleteNode(a);
  g.DetachDeleteNode(b);
}

// Pivot a Pauli interior spider against a Pauli spider owning one boundary.
// The interior spider survives and takes the boundary with its wire toggled.
int64_t boundaryPass(mgp::Graph &g, const std::string &gid, bool gadget) {
  std::vector<std::pair<int64_t, mgp::Node>> nodes;
  for (auto n : g.Nodes()) if (owned(n, gid)) nodes.emplace_back(n.Id().AsInt(), n);
  std::sort(nodes.begin(), nodes.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
  std::unordered_set<int64_t> taken;
  int64_t applied = 0;
  for (auto &[aid, a] : nodes) {
    if (taken.count(aid) || getT(a) != 1 || !isPauli(getPhase(a)) || !interior(a)) continue;
    auto an = nbrs(a);
    bool blocked = false;
    for (auto &[id, item] : an)
      if (taken.count(id) || (gadget && degree(item.second) == 1)) blocked = true;
    if (blocked) continue;
    for (auto &[bid, item] : an) {
      auto b = item.second;
      if (!gadget && !isPauli(getPhase(b))) continue;
      auto bn = nbrs(b);
      std::vector<mgp::Node> boundaries;
      int64_t boundaryType = 0;
      bool valid = true;
      for (auto &[id, neighbor] : bn) {
        if (taken.count(id)) valid = false;
        if (getT(neighbor.second) == 0) {
          boundaries.push_back(neighbor.second);
          boundaryType = relT(neighbor.first);
        } else if (getT(neighbor.second) != 1 || relT(neighbor.first) != 2) valid = false;
      }
      if (!valid || boundaries.size() != 1 || degree(boundaries[0]) != 1
          || (gadget && boundaryType != 1)) continue;
      auto boundary = boundaries[0];
      taken.insert(aid); taken.insert(bid);
      for (auto &[id, neighbor] : an) taken.insert(id);
      for (auto &[id, neighbor] : bn) taken.insert(id);
      bn.erase(boundary.Id().AsInt());
      if (gadget) apply(g, a, b, an, bn, true);
      else applyPivot(g, a, b, an, bn, true);
      auto wire = g.CreateRelationship(a, boundary, "Wire");
      wire.SetProperty("t", mgp::Value(3 - boundaryType));
      wire.SetProperty("graph_id", mgp::Value(gid));
      ++applied;
      break;
    }
  }
  return applied;
}

// Equal open neighborhoods identify nonadjacent twins; equal closed
// neighborhoods identify adjacent twins. Group once instead of joining all pairs.
int64_t supplementarityPass(mgp::Graph &g, const std::string &gid) {
  using Groups = std::map<std::vector<int64_t>, std::vector<mgp::Node>>;
  Groups groups[2];
  for (auto n : g.Nodes()) {
    if (!owned(n, gid) || getT(n) != 1 || !interior(n)) continue;
    double twice = 2 * getPhase(n);
    if (std::fabs(twice - std::round(twice)) < EPS) continue;
    std::vector<int64_t> key;
    for (auto &[id, item] : nbrs(n)) key.push_back(id);
    groups[0][key].push_back(n);
    key.push_back(n.Id().AsInt());
    std::sort(key.begin(), key.end());
    groups[1][key].push_back(n);
  }
  struct Match { mgp::Node a, b; std::vector<mgp::Node> neighbors; bool shift; };
  std::vector<Match> matches;
  std::unordered_set<int64_t> taken;
  for (int type = 0; type < 2; ++type) {
    for (auto &[key, nodes] : groups[type]) {
      if (nodes.size() < 2) continue;
      std::sort(nodes.begin(), nodes.end(), [](auto a, auto b) { return a.Id() < b.Id(); });
      for (size_t i = 0; i < nodes.size(); ++i) {
        auto a = nodes[i];
        if (taken.count(a.Id().AsInt())) continue;
        bool blocked = false;
        for (auto id : key) if (taken.count(id)) blocked = true;
        if (blocked) continue;
        for (size_t j = i + 1; j < nodes.size(); ++j) {
          auto b = nodes[j];
          if (taken.count(b.Id().AsInt())) continue;
          double sum = fmodpos(getPhase(a) + getPhase(b), 2.0);
          bool shift = type == 0 ? std::fabs(sum - 1) < EPS
                                : (sum < EPS || std::fabs(sum - 2) < EPS);
          bool difference = std::fabs(fmodpos(getPhase(a) - getPhase(b), 2.0) - 1) < EPS;
          if (!shift && !difference) continue;
          std::vector<mgp::Node> neighbors;
          for (auto &[id, item] : nbrs(a))
            if (id != b.Id().AsInt()) neighbors.push_back(item.second);
          matches.push_back({a, b, std::move(neighbors), shift});
          taken.insert(a.Id().AsInt()); taken.insert(b.Id().AsInt());
          for (auto id : key) taken.insert(id);
          break;
        }
      }
    }
  }
  for (auto &match : matches) {
    if (match.shift)
      for (auto n : match.neighbors)
        n.SetProperty("phase", mgp::Value(fmodpos(getPhase(n) + 1, 2.0)));
    g.DetachDeleteNode(match.a); g.DetachDeleteNode(match.b);
  }
  return static_cast<int64_t>(matches.size());
}

// Greedy independent matches avoid constructing a Cypher conflict graph.
int64_t cliffordPass(mgp::Graph &g, const std::string &gid, bool pivot) {
  std::vector<std::pair<int64_t, mgp::Node>> nodes;
  for (auto n : g.Nodes()) if (owned(n, gid)) nodes.emplace_back(n.Id().AsInt(), n);
  std::sort(nodes.begin(), nodes.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
  std::unordered_set<int64_t> consumed;
  int64_t applied = 0;
  for (auto &[id, a] : nodes) {
    if (consumed.count(id) || getT(a) != 1) continue;
    auto phase = fmodpos(getPhase(a), 2.0);
    if (pivot ? !isPauli(phase) : (std::fabs(phase-.5) > EPS && std::fabs(phase-1.5) > EPS)) continue;
    if (!interior(a)) continue;
    auto an = nbrs(a);
    bool blocked = false;
    for (auto &[nid, item] : an) if (consumed.count(nid)) blocked = true;
    if (blocked) continue;
    if (!pivot) {
      std::vector<mgp::Node> neighbors;
      for (auto &[nid, item] : an) neighbors.push_back(item.second);
      for (size_t i = 0; i < neighbors.size(); ++i) {
        std::vector<mgp::Node> tail(neighbors.begin()+i+1, neighbors.end());
        toggle(g, {neighbors[i]}, tail, a.GetProperty("graph_id"));
      }
      for (auto n : neighbors) n.SetProperty("phase", mgp::Value(fmodpos(getPhase(n)-phase, 2.0)));
      consumed.insert(id);
      for (auto &[nid, item] : an) consumed.insert(nid);
      g.DetachDeleteNode(a);
      ++applied;
      continue;
    }
    for (auto &[bid, item] : an) {
      auto b = item.second;
      if (!isPauli(getPhase(b)) || !interior(b)) continue;
      auto bn = nbrs(b);
      bool overlap = false;
      for (auto &[nid, ni] : bn) if (consumed.count(nid)) overlap = true;
      if (overlap) continue;
      applyPivot(g, a, b, an, bn);
      consumed.insert(id); consumed.insert(bid);
      for (auto &[nid, ni] : an) consumed.insert(nid);
      for (auto &[nid, ni] : bn) consumed.insert(nid);
      ++applied;
      break;
    }
  }
  return applied;
}
}  // namespace

template<int Rule>
void ScopedRewrite(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  try {
    mgp::Graph graph{memgraph_graph};
    std::string gid(mgp::List(args)[0].ValueString());
    validateScope(graph, gid);
    int64_t total = 0;
    for (int round = 0; round < 10000; ++round) {
      int64_t n = Rule == 5 ? boundaryPass(graph, gid, true)
                  : Rule == 4 ? boundaryPass(graph, gid, false)
                  : Rule == 3 ? supplementarityPass(graph, gid)
                  : Rule == 2 ? onePass(graph, &gid) : cliffordPass(graph, gid, Rule == 1);
      total += n;
      if (!n) {
        mgp::RecordFactory(result).NewRecord().Insert("applied", total);
        return;
      }
    }
    throw std::runtime_error("Native rewrite did not converge in 10000 passes");
  } catch (const std::exception &e) {
    mgp::RecordFactory(result).SetErrorMessage(e.what());
  }
}

void PivotGadget(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  try {
    mgp::Graph graph{memgraph_graph};
    int64_t total = 0;
    while (true) {
      int64_t n = onePass(graph);
      total += n;
      if (n == 0) break;
    }
    auto record = mgp::RecordFactory(result).NewRecord();
    record.Insert("applied", total);
  } catch (const std::exception &e) {
    mgp::RecordFactory(result).SetErrorMessage(e.what());
  }
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::MemoryDispatcherGuard guard{memory};
    mgp::AddProcedure(PivotGadget, "pivot_gadget_fixpoint", mgp::ProcedureType::Write,
                      {}, {mgp::Return("applied", mgp::Type::Int)}, module, memory);
    mgp::AddProcedure(ScopedRewrite<0>, "lcomp", mgp::ProcedureType::Write,
                      {mgp::Parameter("graph_id", mgp::Type::String)},
                      {mgp::Return("applied", mgp::Type::Int)}, module, memory);
    mgp::AddProcedure(ScopedRewrite<1>, "pivot", mgp::ProcedureType::Write,
                      {mgp::Parameter("graph_id", mgp::Type::String)},
                      {mgp::Return("applied", mgp::Type::Int)}, module, memory);
    mgp::AddProcedure(ScopedRewrite<5>, "boundary", mgp::ProcedureType::Write,
                      {mgp::Parameter("graph_id", mgp::Type::String)},
                      {mgp::Return("applied", mgp::Type::Int)}, module, memory);
    mgp::AddProcedure(ScopedRewrite<4>, "pauli_boundary", mgp::ProcedureType::Write,
                      {mgp::Parameter("graph_id", mgp::Type::String)},
                      {mgp::Return("applied", mgp::Type::Int)}, module, memory);
    mgp::AddProcedure(ScopedRewrite<3>, "supplementarity", mgp::ProcedureType::Write,
                      {mgp::Parameter("graph_id", mgp::Type::String)},
                      {mgp::Return("applied", mgp::Type::Int)}, module, memory);
    mgp::AddProcedure(ScopedRewrite<2>, "gadget", mgp::ProcedureType::Write,
                      {mgp::Parameter("graph_id", mgp::Type::String)},
                      {mgp::Return("applied", mgp::Type::Int)}, module, memory);
  } catch (const std::exception &e) {
    return 1;
  }
  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
