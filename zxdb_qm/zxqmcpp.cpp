// In-process C++ pivot_gadget for zxdb (Memgraph query module), ported from
// pyzx's match_pivot_gadget + pivot and from the validated Python module zxqm.
// Runs the whole pivot_gadget fixpoint in one CALL using pyzx's greedy maximal
// disjoint-set matching (mark consumed neighbourhoods), with the heavy O(deg^2)
// edge toggles done natively. Procedure: zxqmcpp.pivot_gadget_fixpoint() -> applied.
#include <mgp.hpp>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <utility>
#include <vector>

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
      if (it != anb.end()) g.DeleteRelationship(it->second);
      else addH(g, a, b, gid);
    }
  }
}
void apply(mgp::Graph &g, mgp::Node &zj, mgp::Node &za, NbrMap &zjn, NbrMap &zan) {
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
  g.DetachDeleteNode(zj);
  g.DetachDeleteNode(za);
}
int64_t onePass(mgp::Graph &g) {
  std::set<int64_t> consumed;
  int64_t applied = 0;
  std::vector<mgp::Node> nodes;
  for (auto n : g.Nodes()) nodes.push_back(n);
  std::sort(nodes.begin(), nodes.end(),
            [](const mgp::Node &a, const mgp::Node &b) { return a.Id().AsInt() < b.Id().AsInt(); });
  for (auto &zj : nodes) {
    int64_t zjid = zj.Id().AsInt();
    if (consumed.count(zjid) || getT(zj) != 1) continue;
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
}  // namespace

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
  } catch (const std::exception &e) {
    return 1;
  }
  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
