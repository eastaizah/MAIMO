import dataclasses
from config import DEFAULT
import channel as ch

print("targets: LS -18.7, edge -22.4, device -19.1 dB")
print("--- edge covariance mismatch sweep ---")
for eps in (0.15, 0.18, 0.19, 0.195, 0.20, 0.21, 0.23):
    p = dataclasses.replace(DEFAULT, nmse_realisations=300,
                            nmse_edge_cov_mismatch=eps)
    r = ch.channel_estimation_nmse(1, p)
    print("eps_edge=%-7g LS %6.2f  edge %6.2f  ideal %6.2f"
          % (eps, r["ls_nmse_db"], r["edge_7b_lora_nmse_db"],
             r["ideal_lmmse_nmse_db"]))
print("--- device covariance mismatch sweep (INT4 per-PRB output) ---")
for eps in (0.0, 0.1, 0.2, 0.3, 0.4, 0.44, 0.5, 0.6):
    p = dataclasses.replace(DEFAULT, nmse_realisations=300,
                            nmse_device_cov_mismatch=eps)
    r = ch.channel_estimation_nmse(1, p)
    print("eps_dev=%-7g device %6.2f" % (eps, r["device_50m_int4_nmse_db"]))
