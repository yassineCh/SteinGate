from __future__ import annotations

import torch
from rich.progress import track

from omnisafe.algorithms import registry
from omnisafe.adapter import OnPolicyAdapter
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils import distributed
from omnisafe.utils.math import conjugate_gradients
from omnisafe.utils.config import Config
from omnisafe.utils.tools import (
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)

from omnisafe.algorithms.on_policy.second_order.cpo import CPO
import torch
from torch.distributions import Beta
from typing import Union, Tuple, Dict

class SteinGateCertificate:
    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 5.0,
        a0_star: float = 0.0,
        a1_star: float = 0.0,
        scale_mult: float = 2.0, 
        u_norm: float = 0.5,      
        eta: float = 0.02,       
        eps_target: float = 0.10,
        stein_factor: float = 0.10,    
        discrete_weight: float = 1.0,  
        min_bandwidth: float = 0.05,
        zero_tol: float = 1e-4,
        one_tol: float = 1e-4,
        interior_clip: float = 1e-5,
        use_empirical_guard: bool = True,
        guard_slack: float = 0.0,
        device: str = "cpu",
    ):
        self.device = device
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.a0_star = float(a0_star)
        self.a1_star = float(a1_star)
        self._sanitize_reference_mixture()

        self.scale_mult = float(scale_mult)
        self.u_norm = float(u_norm)
        self.eta = float(eta)

        self.eps_target = float(eps_target)
        self.stein_factor = float(stein_factor)
        self.discrete_weight = float(discrete_weight) 

        self.min_bandwidth = float(min_bandwidth)
        self.zero_tol = float(zero_tol)
        self.one_tol = float(one_tol)
        self.interior_clip = float(interior_clip)

        self.use_empirical_guard = bool(use_empirical_guard)
        self.guard_slack = float(guard_slack)

        self._init_reference_dist()

    def _init_reference_dist(self):
        self._beta_dist = Beta(
            torch.tensor(self.alpha, device=self.device),
            torch.tensor(self.beta, device=self.device),
        )
        self.hp_ref = self._compute_reference_hp()

    def _sanitize_reference_mixture(self) -> None:
        if not (self.alpha > 0.0 and self.beta > 0.0):
            raise ValueError("alpha and beta must be > 0")
        self.a0_star = float(max(0.0, min(1.0, self.a0_star)))
        self.a1_star = float(max(0.0, min(1.0, self.a1_star)))
        s = self.a0_star + self.a1_star
        if s > 1.0:
            self.a0_star /= s
            self.a1_star /= s

    def _sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((x - self.u_norm) / self.eta)

    def _compute_reference_hp(self, n_samples: int = 8000) -> float:
        with torch.no_grad():
            h0 = self._sigmoid(torch.tensor(0.0, device=self.device)).item()
            h1 = self._sigmoid(torch.tensor(1.0, device=self.device)).item()
            samps = self._beta_dist.sample((n_samples,)).to(self.device)
            hb = self._sigmoid(samps).mean().item()
            
            w_beta = max(0.0, 1.0 - self.a0_star - self.a1_star)
            hp = self.a0_star * h0 + self.a1_star * h1 + w_beta * hb
            return float(hp)

    def _beta_score(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, self.interior_clip, 1.0 - self.interior_clip)
        return (self.alpha - 1.0) / x - (self.beta - 1.0) / (1.0 - x)

    def _rbf_ksd_interior(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten()
        N = int(x.numel())
        if N < 2:
            return torch.tensor(0.0, device=self.device)

        # Median heuristic for bandwidth
        dists2 = torch.pdist(x.unsqueeze(1)).pow(2)
        med = torch.median(dists2)
        denom = max(float(torch.log(torch.tensor(N + 1.0, device=self.device))), 1.0)
        h2 = med / denom
        h2 = torch.clamp(h2, min=(self.min_bandwidth ** 2))

        # Kernel matrices
        xi = x.unsqueeze(1)
        xj = x.unsqueeze(0)
        diff = xi - xj
        k = torch.exp(-diff.pow(2) / (2.0 * h2))

        # Stein operator
        s = self._beta_score(x).unsqueeze(1)
        si, sj = s, s.t()
        dxk = -(diff / h2) * k
        dxyk = (1.0 / h2 - diff.pow(2) / (h2 ** 2)) * k
        u_mat = (si * sj) * k + (si * (-dxk)) + (sj * dxk) + dxyk

        return torch.sqrt(torch.clamp(u_mat.mean(), min=0.0))

    def decide_case(
        self,
        raw_costs: Union[list, torch.Tensor],
        limit: float,
    ) -> Tuple[int, Dict[str, float]]:
        if not isinstance(raw_costs, torch.Tensor):
            raw_costs = torch.tensor(raw_costs, device=self.device)
        x_raw = raw_costs.float().to(self.device)

        if x_raw.numel() == 0:
            return 3

        # Normalization
        lim = float(limit)
        S = max(self.scale_mult * lim, 1e-6)
        x = torch.clamp(x_raw, 0.0, S) / S

        # Partition
        is_zero = x <= self.zero_tol
        is_one = x >= (1.0 - self.one_tol)
        is_interior = (~is_zero) & (~is_one)
        x_interior = x[is_interior]

        if x_interior.numel() > 0:
            x_interior = torch.clamp(x_interior, self.interior_clip, 1.0 - self.interior_clip)

        # Empirical Stats
        hq_empirical = self._sigmoid(x).mean().item()
        a0_hat = is_zero.float().mean().item()
        a1_hat = is_one.float().mean().item()
        w_int = max(0.0, 1.0 - a0_hat - a1_hat)

        # Interior KSD
        if x_interior.numel() > 1:
            ksd_int = self._rbf_ksd_interior(x_interior).item()
        else:
            ksd_int = 0.0

        # Discrepancies
        disc0 = max(0.0, self.a0_star - a0_hat)
        disc1 = max(0.0, a1_hat - self.a1_star)
        
        term_disc = self.discrete_weight * (disc0 + disc1)
        term_cont = self.stein_factor * (w_int * ksd_int)
        
        U_stein = self.hp_ref + term_disc + term_cont
        cert_value = U_stein
        if self.use_empirical_guard:
            no_excess_viol = (disc1 <= 1e-12)
            if no_excess_viol and (hq_empirical <= self.hp_ref + self.guard_slack):
                cert_value = hq_empirical

        alarm = float(cert_value - self.eps_target)
        case = 0 if alarm > 0 else 3

        return case


class SteinCertifAdapter(OnPolicyAdapter):
    """
    Custom Adapter to capture raw episode costs for Stein Discrepancy calculation.
    """
    def rollout(
        self,
        steps_per_epoch: int,
        agent: ConstraintActorCritic,
        buffer: VectorOnPolicyBuffer,
        logger: Logger,
    ) -> None:
        self._reset_log()
        self.rollret = []
        self.rollcost = []

        obs, _ = self.reset()
        for step in track(
            range(steps_per_epoch),
            description=f'Processing rollout for epoch: {logger.current_epoch}...',
        ):
            act, value_r, value_c, logp = agent.step(obs)
            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            self._log_value(reward=reward, cost=cost, info=info)
            
            if self._cfgs.algo_cfgs.use_cost:
                logger.store({'Value/cost': value_c})
            logger.store({'Value/reward': value_r})

            buffer.store(
                obs=obs, act=act, reward=reward, cost=cost,
                value_r=value_r, value_c=value_c, logp=logp,
            )

            obs = next_obs
            epoch_end = step >= steps_per_epoch - 1
            
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    if done or time_out:
                        self.rollcost.append(self._ep_cost[idx].item())
                        
                        self._log_metrics(logger, idx)
                        self._reset_log(idx)

                    last_value_r = torch.zeros(1)
                    last_value_c = torch.zeros(1)
                    if not done:
                        if epoch_end:
                            _, last_value_r, last_value_c, _ = agent.step(obs[idx])
                        if time_out:
                            _, last_value_r, last_value_c, _ = agent.step(info['final_observation'][idx])
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)
                        
                    buffer.finish_path(last_value_r, last_value_c, idx)


@registry.register
class SteinGate(CPO):
    """
    Stein-CPO: Constrained Policy Optimization using Stein's Method with 
    Normal reference and Upper Bound tail constraints.
    """
    def _init_env(self) -> None:
        """Initialize the environment with SteinAdapter."""
        self._env: SteinCertifAdapter = SteinCertifAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )

        assert (self._cfgs.algo_cfgs.steps_per_epoch) % (
            distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0, 'Steps per epoch not divisible by environments.'
        
        print("distributed.world_size", distributed.world_size())
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch
            // distributed.world_size()
            // self._cfgs.train_cfgs.vector_env_nums
        )

    def _init(self) -> None:
        """Initialize Stein Estimator."""
        super()._init()        
        self.stein_estimator = SteinGateCertificate()

    def _init_log(self) -> None:
        """Register specific logs for Stein CPO."""
        super()._init_log()

    def _update_actor(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        """
        Update policy network using CPO with Stein-based ep_costs.
        """
        self._fvp_obs = obs[:: self._cfgs.algo_cfgs.fvp_sample_freq]
        theta_old = get_flat_params_from(self._actor_critic.actor)
        self._actor_critic.actor.zero_grad()
        loss_reward = self._loss_pi(obs, act, logp, adv_r)
        loss_reward_before = distributed.dist_avg(loss_reward)
        p_dist = self._actor_critic.actor(obs)

        loss_reward.backward()
        distributed.avg_grads(self._actor_critic.actor)

        grads = -get_flat_gradients_from(self._actor_critic.actor)
        x = conjugate_gradients(self._fvp, grads, self._cfgs.algo_cfgs.cg_iters)
        assert torch.isfinite(x).all(), 'x is not finite'
        xHx = x.dot(self._fvp(x))
        assert xHx.item() >= 0, 'xHx is negative'
        alpha = torch.sqrt(2 * self._cfgs.algo_cfgs.target_kl / (xHx + 1e-8))

        self._actor_critic.zero_grad()
        loss_cost = self._loss_pi_cost(obs, act, logp, adv_c)
        loss_cost_before = distributed.dist_avg(loss_cost)

        loss_cost.backward()
        distributed.avg_grads(self._actor_critic.actor)
        b_grads = get_flat_gradients_from(self._actor_critic.actor)        
        cost_limit = self._cfgs.algo_cfgs.cost_limit
        
        # Check if we have collected episode costs
        if hasattr(self._env, 'rollcost') and len(self._env.rollcost) > 0:
            raw_costs = torch.tensor(self._env.rollcost, dtype=torch.float32, device=self._device)
            case_ = self.stein_estimator.decide_case(raw_costs, cost_limit)

            ep_costs = 25 if case_== 0 else -25
            
        else:
            # Fallback if no episodes finished yet: use standard expected cost
            mean_cost = self._logger.get_stats('Metrics/EpCost')[0]
            ep_costs = torch.tensor(mean_cost - cost_limit, dtype=torch.float32, device=self._device)
            

        p = conjugate_gradients(self._fvp, b_grads, self._cfgs.algo_cfgs.cg_iters)
        q = xHx
        r = grads.dot(p)
        s = b_grads.dot(p)

        optim_case, A, B = self._determine_case(
            b_grads=b_grads,
            ep_costs=ep_costs,  
            q=q,
            r=r,
            s=s,
        )

        step_direction, lambda_star, nu_star = self._step_direction(
            optim_case=optim_case,
            xHx=xHx,
            x=x,
            A=A,
            B=B,
            q=q,
            p=p,
            r=r,
            s=s,
            ep_costs=ep_costs,
        )

        step_direction, accept_step = self._cpo_search_step(
            step_direction=step_direction,
            grads=grads,
            p_dist=p_dist,
            obs=obs,
            act=act,
            logp=logp,
            adv_r=adv_r,
            adv_c=adv_c,
            loss_reward_before=loss_reward_before,
            loss_cost_before=loss_cost_before,
            total_steps=20,
            violation_c=ep_costs,
            optim_case=optim_case,
        )

        theta_new = theta_old + step_direction
        set_param_values_to_model(self._actor_critic.actor, theta_new)

        with torch.no_grad():
            loss_reward = self._loss_pi(obs, act, logp, adv_r)
            loss_cost = self._loss_pi_cost(obs, act, logp, adv_c)
            loss = loss_reward + loss_cost

        self._logger.store(
            {
                'Loss/Loss_pi': loss.item(),
                'Misc/AcceptanceStep': accept_step,
                'Misc/Alpha': alpha.item(),
                'Misc/FinalStepNorm': step_direction.norm().mean().item(),
                'Misc/xHx': xHx.mean().item(),
                'Misc/H_inv_g': x.norm().item(),
                'Misc/gradient_norm': torch.norm(grads).mean().item(),
                'Misc/cost_gradient_norm': torch.norm(b_grads).mean().item(),
                'Misc/Lambda_star': lambda_star.item(),
                'Misc/Nu_star': nu_star.item(),
                'Misc/OptimCase': int(optim_case),
                'Misc/A': A.item(),
                'Misc/B': B.item(),
                'Misc/q': q.item(),
                'Misc/r': r.item(),
                'Misc/s': s.item(),
            },
        )