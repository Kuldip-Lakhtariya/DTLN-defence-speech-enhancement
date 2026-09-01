from scipy.signal import correlate
lag = np.argmax(correlate(clean_signal, enhanced_signal)) - (len(enhanced_signal) - 1)
print("Best-fit lag (samples):", lag)