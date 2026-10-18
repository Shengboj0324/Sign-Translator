Superseded attempt, preserved as history.

This plan used positional rotating validation (fold i validates on component i+1).
On the real mapping the dominant component c02 (signers 5 and 8, 26,806 rows, 86%)
became the validation set of fold 1, leaving 3,035 training rows for no
statistical benefit. The replacement in `../w1-fold-plan-2026-10-17/` chooses each
fold's validation component by a declared exact integer argmin rule instead.
Component membership and leakage freedom are identical in both plans.
