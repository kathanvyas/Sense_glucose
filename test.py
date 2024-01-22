import neurokit2 as nk
import matplotlib.pyplot as plt

ecg = nk.ecg_simulate(duration=6, sampling_rate=1000)
print(ecg.shape)
_, rpeaks = nk.ecg_peaks(ecg)
print(rpeaks)
signals, waves = nk.ecg_delineate(ecg, rpeaks, sampling_rate=1000)
print(signals)
print(waves)
print(signals.iloc[waves["ECG_P_Peaks"][0]])
print(len(waves["ECG_P_Peaks"]))
nk.events_plot([waves["ECG_P_Peaks"], waves["ECG_T_Peaks"]], ecg)
plt.savefig("test.png")
# cardiac_phase = nk.ecg_phase(ecg_cleaned=ecg, rpeaks=rpeaks,
#                           delineate_info=waves, sampling_rate=1000)


# _, ax = plt.subplots(nrows=2)

# ax[0].plot(nk.rescale(ecg), label="ECG", color="red", alpha=0.3)


# ax[0].plot(cardiac_phase["ECG_Phase_Atrial"], label="Atrial Phase", color="orange")

# ax[0].plot(cardiac_phase["ECG_Phase_Completion_Atrial"],
#           label="Atrial Phase Completion", linestyle="dotted")

# ax[0].legend(loc="upper right")

# ax[1].plot(nk.rescale(ecg), label="ECG", color="red", alpha=0.3)

# ax[1].plot(cardiac_phase["ECG_Phase_Ventricular"], label="Ventricular Phase", color="green")

# ax[1].plot(cardiac_phase["ECG_Phase_Completion_Ventricular"],
#        label="Ventricular Phase Completion", linestyle="dotted")

# ax[1].legend(loc="upper right")

# plt.savefig("test.png")