# SteinGate

This repository contains the code required to integrate **SteinGate** into an
[OmniSafe](https://github.com/PKU-Alignment/omnisafe) installation.

SteinGate is the algorithm introduced in our UAI paper:

**SteinGate: Tail-Sensitive Safe Reinforcement Learning via Stein Discrepancy**

SteinGate addresses safe reinforcement learning settings where constraining only
the expected cumulative cost can miss rare but severe tail events. Instead of
directly fitting the tail of the rollout cost distribution, SteinGate uses a
Kernelized Stein Discrepancy (KSD)-based distributional safety certificate to
check whether recent rollout costs remain consistent with a designated safe
reference distribution.

The certificate acts as a gate during training:

- when the certificate is satisfied, the policy update follows the
  reward-improving direction;
- when the certificate is violated, the update switches to safety-recovery
  behavior and prioritizes cost reduction.

This implementation provides SteinGate as an OmniSafe-compatible on-policy
algorithm using CPO as the base optimization engine.

---

## Files

This repository provides two integration files:

```text
steingate.py
SteinGate.yaml
```

---

## Integration with OmniSafe

First, install OmniSafe following the official OmniSafe installation instructions.

Then copy the provided files into your OmniSafe source tree:

```text
steingate.py
→ omnisafe/algorithms/on_policy/steingate.py

SteinGate.yaml
→ omnisafe/configs/on_policy/steingate.yaml
```

Next, import and register `SteinGate` in:

```text
omnisafe/algorithms/__init__.py
```

Follow the same pattern used by the other OmniSafe on-policy algorithms.

---

## Usage

After integration, run SteinGate from the `omnisafe/examples` directory using
the standard OmniSafe example training script:

```bash
cd omnisafe/examples
python train_policy.py --algo SteinGate --env-id SafetyPointGoal1-v0
```

Adjust the environment name, seed, device, and training configuration in
`SteinGate.yaml` as needed.

---

## Configuration

The default configuration is provided in `SteinGate.yaml`.

The implementation includes:

- standard OmniSafe on-policy training options;
- CPO-style second-order update parameters;
- SteinGate certificate parameters through the `SteinGateCertificate` class;

The default certificate uses a hybrid reference model for normalized clipped
episode costs. Boundary atoms handle mass at 0 and 1, while a continuous
interior reference distribution, by default a Beta distribution, is used for
interior costs.

