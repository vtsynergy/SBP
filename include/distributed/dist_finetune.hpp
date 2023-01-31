#include "distributed/dist_blockmodel_triplet.hpp"
#include "distributed/two_hop_blockmodel.hpp"

namespace finetune::dist {

/// Updates `blockmodel` for one membership update contained in `membership`.
void async_move(const Membership &membership, const Graph &graph, TwoHopBlockmodel &blockmodel);

/// Runs the Asynchronous Gibbs algorithm in a distributed fashion using MPI.
TwoHopBlockmodel &asynchronous_gibbs(TwoHopBlockmodel &blockmodel, Graph &graph, DistBlockmodelTriplet &blockmodels);

/// If the average of the last 3 delta entropies is < threshold * initial_entropy, stop the algorithm.
bool early_stop(int iteration, DistBlockmodelTriplet &blockmodels, double initial_entropy,
                std::vector<double> &delta_entropies);

/// Runs the hybrid MCMC algorithm in a distributed fashion using MPI.
TwoHopBlockmodel &hybrid_mcmc(TwoHopBlockmodel &blockmodel, Graph &graph, DistBlockmodelTriplet &blockmodels);

/// Runs the Metropolis Hastings algorithm in a distributed fashion using MPI.
TwoHopBlockmodel &metropolis_hastings(TwoHopBlockmodel &blockmodel, Graph &graph, DistBlockmodelTriplet &blockmodels);

/// Runs one iteration of the Metropolis-Hastings algorithm. Returns the accepted vertex moves.
std::vector<Membership> metropolis_hastings_iteration(TwoHopBlockmodel &blockmodel, Graph &graph,
                                                      const std::vector<int> &active_set = std::vector<int>());

/// Proposes an asynchronous Gibbs move in a distributed setting.
VertexMove propose_gibbs_move(const TwoHopBlockmodel &blockmodel, int vertex, const Graph &graph);

/// Proposes a metropolis hastings move in a distributed setting.
VertexMove propose_mh_move(TwoHopBlockmodel &blockmodel, int vertex, const Graph &graph);

}  // namespace finetune::dist