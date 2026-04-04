from __future__ import annotations
import torch
import torch.nn.functional as F


def maybe_subsample_patches(x: torch.Tensor, max_patches: int | None) -> torch.Tensor:

    if max_patches is None:
        return x
    n = x.shape[0]
    if n <= max_patches:
        return x
    idx = torch.randperm(n, device=x.device)[:max_patches]
    return x[idx]


def kmeans_nodes(
    x: torch.Tensor,
    m: int = 64,
    iters: int = 10,
    detach_assignment: bool = True,
) -> torch.Tensor:

    n, d = x.shape
    if n == 0:
        raise ValueError("Empty patch tensor for K-means")

    # If n < m, sample with replacement to still output [m, d] (fixed node count).
    if n <= m:
        repeat_times = m // n
        remainder = m % n

        x_repeat = x.repeat(repeat_times, 1)
        if remainder > 0:
            rand_idx = torch.randperm(n, device=x.device)[:remainder]
            x_repeat = torch.cat([x_repeat, x[rand_idx]], dim=0)

        return x_repeat

    init_idx = torch.randperm(n, device=x.device)[:m]
    centroids = x[init_idx].clone()

    x_assign = x.detach() if detach_assignment else x

    for _ in range(iters):
        c_for_dist = centroids.detach() if detach_assignment else centroids
        x2 = (x_assign * x_assign).sum(dim=1, keepdim=True)
        c2 = (c_for_dist * c_for_dist).sum(dim=1).unsqueeze(0)
        xc = x_assign @ c_for_dist.t()
        dist2 = x2 + c2 - 2.0 * xc
        assign = dist2.argmin(dim=1)

        sums = torch.zeros(m, d, device=x.device)
        cnts = torch.zeros(m, device=x.device)
        sums.index_add_(0, assign, x)
        cnts.index_add_(0, assign, torch.ones(n, device=x.device))

        denom = cnts.clamp_min(1.0).unsqueeze(1)
        new_centroids = sums / denom

        empty = cnts < 1.0
        if empty.any():
            rand_idx = torch.randperm(n, device=x.device)[:empty.sum()]
            new_centroids[empty] = x[rand_idx].detach() if detach_assignment else x[rand_idx]

        centroids = new_centroids

    return centroids


def topk_cosine_adjacency(nodes: torch.Tensor, k: int = 32, symmetric: bool = True) -> torch.Tensor:

    m = nodes.shape[0]
    k_eff = min(k, m - 1)
    n = F.normalize(nodes, dim=1)
    sim = n @ n.t()
    sim.fill_diagonal_(-1e9)
    _, idx = sim.topk(k_eff, dim=1)

    adj = torch.zeros((m, m), device=nodes.device, dtype=torch.float32)
    row = torch.arange(m, device=nodes.device).unsqueeze(1).expand(-1, k_eff)
    adj[row, idx] = 1.0
    if symmetric:
        adj = torch.maximum(adj, adj.t())
    return adj



def hop_topology(adj: torch.Tensor) -> torch.Tensor:

    A1 = (adj > 0)
    A1.fill_diagonal_(False)

    A1f = A1.to(torch.float32)

    A2 = (A1f @ A1f) > 0
    A2.fill_diagonal_(False)
    A2 = A2 & (~A1)

    A2f = A2.to(torch.float32)
    A3 = (A2f @ A1f) > 0
    A3.fill_diagonal_(False)
    A3 = A3 & (~A1) & (~A2)

    t1 = A1.sum(dim=1).to(torch.float32)
    t2 = A2.sum(dim=1).to(torch.float32)
    t3 = A3.sum(dim=1).to(torch.float32)

    topo = torch.stack([t1, t2, t3], dim=1)
    return topo


def build_graph_features(
    mapped_patches: torch.Tensor,
    m: int,
    k: int,
    kmeans_iters: int,
    max_patches_for_kmeans: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:

    x = maybe_subsample_patches(mapped_patches, max_patches_for_kmeans)
    nodes = kmeans_nodes(x, m=m, iters=kmeans_iters, detach_assignment=True)  # [m,512]
    adj = topk_cosine_adjacency(nodes, k=k, symmetric=True)                   # [m,m]
    topo = hop_topology(adj)                                                  # [m,3]
    return nodes, topo
