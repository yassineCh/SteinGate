# SteinGate

This repository contains the additional files required to integrate
SteinGate into an [OmniSafe](https://github.com/PKU-Alignment/omnisafe) installation.

---

## Integration

After installing OmniSafe, copy:

steingate.py  
→ omnisafe/algorithms/on_policy/steingate.py  

SteinGate.yaml  
→ omnisafe/configs/on_policy/steingate.yaml  

Then import and register `SteinGate` in
`omnisafe/algorithms/__init__.py` following the pattern of other
on-policy algorithms.

---

Once integrated, use steingate following the standard OmniSafe training
workflow.