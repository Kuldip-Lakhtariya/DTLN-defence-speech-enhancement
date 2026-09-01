import os

clean_dir = "data/raw/clean_testset_wav"
noisy_dir = "data/raw/noisy_testset_wav"

clean_files = set(os.listdir(clean_dir))
noisy_files = set(os.listdir(noisy_dir))

print(f"clean_testset_wav has {len(clean_files)} files")
print(f"noisy_testset_wav has {len(noisy_files)} files")

only_in_clean = sorted(clean_files - noisy_files)
only_in_noisy = sorted(noisy_files - clean_files)

print(f"\nIn clean but NOT in noisy ({len(only_in_clean)}):")
for f in only_in_clean[:20]:
    print(" ", f)

print(f"\nIn noisy but NOT in clean ({len(only_in_noisy)}):")
for f in only_in_noisy[:20]:
    print(" ", f)